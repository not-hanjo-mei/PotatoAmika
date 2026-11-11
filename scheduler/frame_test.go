package main

import "testing"

func TestSurfaceFramePlain(t *testing.T) {
	if _, framed, _ := surfaceFrame("Hello there. How are you?"); framed {
		t.Error("plain reply misdetected as frame")
	}
	if got := surfaceText("Hello there. How are you?"); got != "Hello there. How are you?" {
		t.Errorf("surfaceText plain = %q", got)
	}
}

func TestSurfaceFrameFull(t *testing.T) {
	in := "status: ok\ntone: casual_neutral\nmode: say\nreply: I'm Sakurai Mika."
	reply, framed, has := surfaceFrame(in)
	if !framed || !has {
		t.Fatalf("frame not detected: framed=%v has=%v", framed, has)
	}
	if reply != "I'm Sakurai Mika." {
		t.Errorf("reply = %q", reply)
	}
	if got := surfaceText(in); got != "I'm Sakurai Mika." {
		t.Errorf("surfaceText = %q", got)
	}
}

func TestSurfaceFrameMultilineReply(t *testing.T) {
	in := "status: not-for-me\ntone: serious_cold\nmode: say+think\nreply: First line.\nSecond line stays."
	reply, framed, has := surfaceFrame(in)
	if !framed || !has || reply != "First line.\nSecond line stays." {
		t.Errorf("framed=%v has=%v reply=%q", framed, has, reply)
	}
}

func TestSurfaceFramePending(t *testing.T) {
	// Header arrived, reply field not yet: framed but no reply, and the
	// streaming guard must hold everything back.
	_, framed, has := surfaceFrame("status: ok\ntone: casual_neutral\nmode: say")
	if !framed || has {
		t.Errorf("framed=%v has=%v, want true/false", framed, has)
	}
	if got := releasableText("status: ok\ntone: casual_neutral\nmode: say"); got != "" {
		t.Errorf("releasableText pending frame = %q, want empty", got)
	}
}

func TestSurfaceFrameStreamingPrefixes(t *testing.T) {
	full := "status: ok\ntone: casual_neutral\nmode: say\nreply: I'm Sakurai Mika. Nice to meet you."
	want := "I'm Sakurai Mika. Nice to meet you."
	// The released text must grow monotonically and stay a prefix of the
	// final reply for every stream prefix.
	prev := ""
	for i := 0; i <= len(full); i++ {
		got := releasableText(full[:i])
		if len(got) < len(prev) || (got != "" && !hasPrefix(want, got)) {
			t.Fatalf("prefix %d: released %q after %q, want prefix of %q", i, got, prev, want)
		}
		prev = got
	}
	if prev != want {
		t.Errorf("final released = %q, want %q", prev, want)
	}
	if got := surfaceText(full); got != want {
		t.Errorf("surfaceText = %q, want %q", got, want)
	}
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func TestSurfaceTextThinkThenFrame(t *testing.T) {
	in := "<think></think>status: ok\ntone: happy\nmode: say\nreply: Yay, that's great!"
	if got := surfaceText(in); got != "Yay, that's great!" {
		t.Errorf("surfaceText = %q", got)
	}
	if got := releasableText(in); got != "Yay, that's great!" {
		t.Errorf("releasableText = %q", got)
	}
}

func TestReleasablePlainUnchanged(t *testing.T) {
	// Long marker-free plain text keeps the historical behavior.
	long := ""
	for len(long) < 500 {
		long += "Just a normal sentence here. "
	}
	if got := releasableText(long); got != long {
		t.Errorf("long plain: got %q", got)
	}
	// Short plain text is still held back (think-leak guard).
	if got := releasableText("Hi."); got != "" {
		t.Errorf("short plain: got %q, want empty", got)
	}
}
