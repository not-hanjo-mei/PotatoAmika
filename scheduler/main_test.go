package main

import "testing"

func TestStripThink(t *testing.T) {
	cases := []struct{ in, want string }{
		{"<think>plan</think>Hello there.", "Hello there."},
		{"Okay, the user said hi, I should greet back warmly.</think>Tch. Hi.", "Tch. Hi."},
		{"\n\nJust a normal reply.", "Just a normal reply."},
		{"<think>never closed", ""},
	}
	for _, c := range cases {
		if got := stripThink(c.in); got != c.want {
			t.Errorf("stripThink(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSplitter(t *testing.T) {
	sp := &sentenceSplitter{}
	got := sp.Feed("Hi there. How are")
	got = append(got, sp.Feed(" you? Fine... ok 3.5 is a number. Bye")...)
	got = append(got, sp.Flush()...)
	want := []string{"Hi there.", "How are you?", "Fine...", "ok 3.5 is a number.", "Bye"}
	if len(got) != len(want) {
		t.Fatalf("splitter got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("sentence %d: %q != %q", i, got[i], want[i])
		}
	}
}

func TestParseFactsObjectAndList(t *testing.T) {
	obj := "```json\n{\"personal\": [\"The user lives in Portland\", \"The user is allergic to shellfish\"], \"preferences\": [\"The user prefers tea\"]}\n```"
	facts := parseFacts(obj)
	if len(facts) != 3 {
		t.Fatalf("object facts: got %v", facts)
	}
	lst := `["Fact one", "Fact two"]`
	if facts := parseFacts(lst); len(facts) != 2 {
		t.Fatalf("list facts: got %v", facts)
	}
	if facts := parseFacts("{}"); len(facts) != 0 {
		t.Fatalf("empty facts: got %v", facts)
	}
}

func TestFactIDDedupe(t *testing.T) {
	if factID("The user  lives in Portland") != factID("the user lives in portland") {
		t.Error("fact IDs should be normalization-insensitive")
	}
}

func TestEstTokens(t *testing.T) {
	if estTokens("") != 0 {
		t.Error("empty")
	}
	if estTokens("你好世界") != 4 {
		t.Error("CJK count")
	}
}
