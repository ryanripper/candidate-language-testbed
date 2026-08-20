"""Unit tests for the synthetic corpus generator.

The shipped corpus is hash-pinned in ws0-harness/seal_manifest.json, so
these tests never regenerate or overwrite it. They verify the generative
assumptions the whole testbed rests on: determinism under the pinned seed,
ideology bounds and party ordering, topic-mix validity, the
framing-encodes-ideology logistic link, and retweet-source proximity.
"""

import random


class TestDeterminism:
    def test_roster_reproduces_shipped_counts(self, fresh_gen):
        """Under SEED = 20260719 the roster must match the sealed corpus:
        910 candidates, 447 R / 433 D / 30 I, 841 House / 69 Senate."""
        roster = fresh_gen.build_roster()
        assert len(roster) == 910
        parties = [c["party"] for c in roster]
        assert parties.count("R") == 447
        assert parties.count("D") == 433
        assert parties.count("I") == 30
        chambers = [c["chamber"] for c in roster]
        assert chambers.count("House") == 841
        assert chambers.count("Senate") == 69

    def test_roster_is_deterministic(self, gen):
        gen.rng = random.Random(gen.SEED)
        first = gen.build_roster()
        gen.rng = random.Random(gen.SEED)
        second = gen.build_roster()
        assert first == second


class TestRosterInvariants:
    def test_ids_names_handles_unique(self, fresh_gen):
        roster = fresh_gen.build_roster()
        for key in ("candidate_id", "candidate_name", "handle"):
            values = [c[key] for c in roster]
            assert len(values) == len(set(values)), f"duplicate {key}"

    def test_ideology_bounded_and_party_ordered(self, fresh_gen):
        roster = fresh_gen.build_roster()
        assert all(-1.0 <= c["true_ideology"] <= 1.0 for c in roster)
        def mean(party):
            members = [c["true_ideology"] for c in roster if c["party"] == party]
            return sum(members) / len(members)

        assert mean("D") < -0.4
        assert mean("R") > 0.4
        assert abs(mean("I")) < 0.3

    def test_handles_start_with_at(self, fresh_gen):
        roster = fresh_gen.build_roster()
        assert all(c["handle"].startswith("@") for c in roster)


class TestDraws:
    def test_draw_ideology_bounded(self, fresh_gen):
        for party in ("D", "R", "I"):
            draws = [fresh_gen.draw_ideology(party) for _ in range(2000)]
            assert all(-1.0 <= x <= 1.0 for x in draws)

    def test_topic_mix_is_a_distribution(self, fresh_gen):
        for ideology in (-0.9, 0.0, 0.9):
            mix = fresh_gen.draw_topic_mix(ideology)
            assert set(mix) == set(fresh_gen.TOPICS)
            assert abs(sum(mix.values()) - 1.0) < 1e-9
            assert all(v >= 0 for v in mix.values())

    def test_topic_tilt_direction(self, fresh_gen):
        """Averaged over many draws, conservatives emphasize immigration
        more and climate less than liberals (the planted tilt)."""
        n = 3000
        lib = {t: 0.0 for t in fresh_gen.TOPICS}
        con = {t: 0.0 for t in fresh_gen.TOPICS}
        for _ in range(n):
            for t, v in fresh_gen.draw_topic_mix(-0.9).items():
                lib[t] += v / n
            for t, v in fresh_gen.draw_topic_mix(0.9).items():
                con[t] += v / n
        assert con["immigration"] > lib["immigration"]
        assert lib["climate_energy"] > con["climate_energy"]


class TestFramingLink:
    def test_lean_probs_sum_to_one(self, gen):
        for ideology in (-1.0, -0.3, 0.0, 0.4, 1.0):
            pl, pn, pc = gen.lean_probs(ideology)
            assert abs(pl + pn + pc - 1.0) < 1e-12
            assert min(pl, pn, pc) >= 0

    def test_p_con_monotone_in_ideology(self, gen):
        p_cons = [gen.lean_probs(x)[2] for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        assert p_cons == sorted(p_cons)

    def test_extremes_pick_matching_lean(self, fresh_gen):
        n = 2000
        con_picks = [fresh_gen.pick_lean(0.95) for _ in range(n)]
        lib_picks = [fresh_gen.pick_lean(-0.95) for _ in range(n)]
        assert con_picks.count("con") > con_picks.count("lib") * 5
        assert lib_picks.count("lib") > lib_picks.count("con") * 5


class TestTextAssembly:
    def test_fill_leaves_no_placeholders(self, fresh_gen):
        template = "Join us in {city} at {time} on {holiday} — {num} strong in {state}!"
        for _ in range(50):
            out = fresh_gen.fill(template, "OH")
            assert "{" not in out and "}" not in out
            assert "OH" in out

    def test_original_tweet_shape(self, fresh_gen):
        cand = {"true_ideology": 0.6, "state": "TX",
                "topic_mix": {"economy": 0.5, "campaign_logistics": 0.5}}
        for _ in range(100):
            topic, lean, text = fresh_gen.gen_original_tweet(cand)
            assert topic in fresh_gen.TOPICS
            assert lean in ("lib", "neu", "con")
            assert text and "{" not in text
            if topic == "campaign_logistics":
                assert lean == "neu"

    def test_retweet_shape_and_source(self, fresh_gen):
        handles = {s["handle"] for s in fresh_gen.RT_SOURCES}
        cand = {"true_ideology": 0.0}
        for _ in range(100):
            src, text = fresh_gen.gen_retweet(cand)
            assert src in handles
            assert text.startswith(f"RT {src}: ")

    def test_retweets_prefer_nearby_sources(self, fresh_gen):
        """The retweets-as-endorsed-speech mechanism: a strongly
        conservative candidate's retweet sources average right of a
        strongly liberal candidate's."""
        ideo = {s["handle"]: s["ideology"] for s in fresh_gen.RT_SOURCES}
        n = 1500
        con_mean = sum(
            ideo[fresh_gen.gen_retweet({"true_ideology": 0.9})[0]] for _ in range(n)
        ) / n
        lib_mean = sum(
            ideo[fresh_gen.gen_retweet({"true_ideology": -0.9})[0]] for _ in range(n)
        ) / n
        assert con_mean > 0.4
        assert lib_mean < -0.4
