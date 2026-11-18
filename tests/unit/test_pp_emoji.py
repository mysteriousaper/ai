"""Emoji and emoticon mapping, emoticon spacing (brief §4.3)."""

from engine.preprocessing.emoji_map import find_emojis, find_emoticons, space_emoticons


class TestEmoji:
    def test_named(self) -> None:
        hits = find_emojis("great 😀 day")
        assert len(hits) == 1
        assert hits[0].char == "😀"
        assert hits[0].name == "grinning_face"
        assert hits[0].start == 6

    def test_skin_tone_and_variation_selector(self) -> None:
        hits = find_emojis("👍🏽❤️")
        assert len(hits) == 2
        assert all(h.name and ":" not in h.name for h in hits)

    def test_none(self) -> None:
        assert find_emojis("plain text :)") == ()


class TestEmoticons:
    def test_basic(self) -> None:
        names = [h.name for h in find_emoticons("hi :) bye :( lol :D ;) </3 <3")]
        assert names == ["smile", "frown", "laugh", "wink", "broken_heart", "heart"]

    def test_attached_to_word(self) -> None:
        hits = find_emoticons("happy:)")
        assert [(h.text, h.start) for h in hits] == [(":)", 5)]

    def test_url_is_not_emoticon(self) -> None:
        assert find_emoticons("see https://example.com/x") == ()

    def test_letter_emoticon_needs_boundary(self) -> None:
        assert find_emoticons("Dxd :Doc") == ()
        assert [h.name for h in find_emoticons("ok xD")] == ["laugh"]

    def test_longest_match(self) -> None:
        hits = find_emoticons(":-) :'(")
        assert [(h.text, h.name) for h in hits] == [(":-)", "smile"), (":'(", "cry")]

    def test_parenthesised_number_is_not_emoticon(self) -> None:
        assert find_emoticons("(item 8) and (see 3)") == ()


class TestSpacing:
    def test_emoticon_is_spaced(self) -> None:
        assert space_emoticons("happy:)ok") == "happy :) ok"

    def test_already_spaced_unchanged(self) -> None:
        assert space_emoticons("happy :) ok") == "happy :) ok"

    def test_no_emoticons_unchanged(self) -> None:
        assert space_emoticons("plain") == "plain"
