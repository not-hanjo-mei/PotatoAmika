package main

import (
	"bufio"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
	"sync"
)

// Minimal RFC 6455 frame codec shared by the server and the client.
// Ported from adapter.js (WsParser / wsEncode / wsAcceptKey) with the
// same fragmentation and control-frame behaviour.

func wsAcceptKey(key string) string {
	h := sha1.New()
	h.Write([]byte(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func wsEncode(payload []byte, opcode byte, mask bool) []byte {
	var head []byte
	switch n := len(payload); {
	case n < 126:
		head = []byte{0, byte(n)}
	case n < 65536:
		head = make([]byte, 4)
		head[1] = 126
		binary.BigEndian.PutUint16(head[2:], uint16(n))
	default:
		head = make([]byte, 10)
		head[1] = 127
		binary.BigEndian.PutUint64(head[2:], uint64(n))
	}
	head[0] = 0x80 | opcode
	if !mask {
		return append(head, payload...)
	}
	head[1] |= 0x80
	key := make([]byte, 4)
	_, _ = rand.Read(key)
	masked := make([]byte, len(payload))
	for i := range payload {
		masked[i] = payload[i] ^ key[i%4]
	}
	out := append(head, key...)
	return append(out, masked...)
}

type wsParser struct {
	buf       []byte
	frag      [][]byte
	fragOp    byte
	onMessage func(string)
	onBinary  func([]byte)
	onClose   func()
	onPing    func([]byte)
}

func (p *wsParser) push(chunk []byte) {
	p.buf = append(p.buf, chunk...)
	for {
		if len(p.buf) < 2 {
			return
		}
		b0, b1 := p.buf[0], p.buf[1]
		fin := b0&0x80 != 0
		op := b0 & 0x0f
		masked := b1&0x80 != 0
		length := uint64(b1 & 0x7f)
		off := 2
		switch length {
		case 126:
			if len(p.buf) < 4 {
				return
			}
			length = uint64(binary.BigEndian.Uint16(p.buf[2:]))
			off = 4
		case 127:
			if len(p.buf) < 10 {
				return
			}
			length = binary.BigEndian.Uint64(p.buf[2:])
			off = 10
		}
		maskLen := 0
		if masked {
			maskLen = 4
		}
		if uint64(len(p.buf)) < uint64(off+maskLen)+length {
			return
		}
		total := off + maskLen + int(length)
		payload := p.buf[off+maskLen : total]
		if masked {
			key := p.buf[off : off+4]
			cp := make([]byte, len(payload))
			copy(cp, payload)
			for i := range cp {
				cp[i] ^= key[i%4]
			}
			payload = cp
		}
		p.buf = p.buf[total:]
		switch op {
		case 8:
			if p.onClose != nil {
				p.onClose()
			}
			return
		case 9:
			if p.onPing != nil {
				p.onPing(payload)
			}
			continue
		case 10:
			continue
		}
		if op == 0 {
			if p.frag != nil {
				p.frag = append(p.frag, payload)
			}
		} else {
			p.frag = [][]byte{payload}
			p.fragOp = op
		}
		if fin && p.frag != nil {
			full := concatBytes(p.frag)
			fop := p.fragOp
			p.frag = nil
			if fop == 2 {
				if p.onBinary != nil {
					p.onBinary(full)
				}
			} else if p.onMessage != nil {
				p.onMessage(string(full))
			}
		}
	}
}

func concatBytes(parts [][]byte) []byte {
	n := 0
	for _, b := range parts {
		n += len(b)
	}
	out := make([]byte, 0, n)
	for _, b := range parts {
		out = append(out, b...)
	}
	return out
}

// WebSocket client (to the ASR gateway), mirroring adapter.js wsConnect.
type wsClient struct {
	conn   net.Conn
	closed bool
	mu     sync.Mutex

	onMessage func(string)
	onBinary  func([]byte)
	onClose   func()
}

func (c *wsClient) sendText(s string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed {
		_, _ = c.conn.Write(wsEncode([]byte(s), 1, true))
	}
}

func (c *wsClient) sendBinary(b []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed {
		_, _ = c.conn.Write(wsEncode(b, 2, true))
	}
}

func (c *wsClient) close() {
	c.closed = true
	_ = c.conn.Close()
}

func (c *wsClient) fireClose() {
	c.mu.Lock()
	already := c.closed
	c.closed = true
	c.mu.Unlock()
	if !already && c.onClose != nil {
		c.onClose()
	}
}

func wsConnect(rawurl string) (*wsClient, error) {
	u, err := url.Parse(rawurl)
	if err != nil {
		return nil, err
	}
	host := u.Hostname()
	port := u.Port()
	if port == "" {
		port = "80"
	}
	conn, err := net.Dial("tcp", net.JoinHostPort(host, port))
	if err != nil {
		return nil, err
	}
	keyBytes := make([]byte, 16)
	_, _ = rand.Read(keyBytes)
	key := base64.StdEncoding.EncodeToString(keyBytes)
	path := u.EscapedPath()
	if path == "" {
		path = "/"
	}
	if u.RawQuery != "" {
		path += "?" + u.RawQuery
	}
	_, err = fmt.Fprintf(conn, "GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n",
		path, u.Host, key)
	if err != nil {
		conn.Close()
		return nil, err
	}
	client := &wsClient{conn: conn}
	br := bufio.NewReader(conn)
	// Read the handshake response header block.
	statusLine, err := br.ReadString('\n')
	if err != nil {
		conn.Close()
		return nil, err
	}
	if !strings.Contains(statusLine, "101") {
		conn.Close()
		return nil, errors.New("ASR gateway handshake: " + strings.TrimRight(statusLine, "\r\n"))
	}
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			conn.Close()
			return nil, err
		}
		if line == "\r\n" || line == "\n" {
			break
		}
	}
	parser := &wsParser{
		onMessage: func(s string) {
			if client.onMessage != nil {
				client.onMessage(s)
			}
		},
		onBinary: func(b []byte) {
			if client.onBinary != nil {
				client.onBinary(b)
			}
		},
		onClose: func() { client.fireClose() },
	}
	parser.onPing = func(pl []byte) {
		client.mu.Lock()
		if !client.closed {
			_, _ = conn.Write(wsEncode(pl, 10, true))
		}
		client.mu.Unlock()
	}
	go func() {
		buf := make([]byte, 32*1024)
		if n := br.Buffered(); n > 0 {
			rest, _ := br.Peek(n)
			parser.push(append([]byte{}, rest...))
			_, _ = br.Discard(n)
		}
		for {
			n, err := br.Read(buf)
			if n > 0 {
				parser.push(buf[:n])
			}
			if err != nil {
				client.fireClose()
				return
			}
		}
	}()
	return client, nil
}
