#!/usr/bin/env python3
"""
Synthetic congressional-candidate tweet corpus generator.

Produces a single flat CSV of fictional candidates and their tweets/retweets,
with PLANTED latent structure so embedding/PCA/topic methods have known ground
truth to recover:

  * Each candidate has a true_ideology score in [-1, 1] (liberal -> conservative),
    drawn from party-conditional distributions.
  * Each candidate has a Dirichlet topic mix over 10 policy topics (plus a
    "campaign logistics" topic uncorrelated with ideology).
  * Tweet text is assembled from topic-specific phrase banks; the probability of
    drawing liberal- vs conservative-coded framing is a logistic function of
    true_ideology, so lexical choice encodes the latent score.
  * Retweets are drawn from a pool of fictional org/pundit accounts, each with
    its own ideology; candidates preferentially retweet nearby accounts
    (retweets-as-endorsed-speech, matching the original research design).

Everything (names, handles, orgs) is fictional. Deterministic under SEED.
"""

import csv
import gzip
import math
import os
import random
from datetime import datetime, timedelta

SEED = 20260719
rng = random.Random(SEED)

# ----------------------------------------------------------------------------
# Roster construction
# ----------------------------------------------------------------------------

STATES = {
    "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
    "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
    "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
    "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
    "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
    "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WV": 2,
    "WI": 8, "WY": 1,
}
SENATE_2022 = ["AL", "AK", "AR", "CO", "DE", "GA", "ID", "IL", "IA", "KS", "KY",
               "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM",
               "NC", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY"]

FIRST_NAMES = [
    "Avery", "Jordan", "Marisol", "Deshawn", "Priya", "Colton", "Ingrid",
    "Tobias", "Renata", "Marcus", "Elena", "Harlan", "Yolanda", "Pete",
    "Cassidy", "Omar", "Lucille", "Grant", "Noemi", "Walker", "Simone",
    "Dexter", "Paloma", "Russell", "Tamika", "Boyd", "Celeste", "Emmett",
    "Farrah", "Gideon", "Hattie", "Ivan", "Junia", "Kendall", "Lorenzo",
    "Maeve", "Nolan", "Opal", "Quincy", "Rosalind", "Sterling", "Thea",
    "Ulysses", "Vera", "Wendell", "Ximena", "York", "Zelda", "Anders",
    "Bettina", "Cormac", "Delphine", "Ezra", "Flora", "Gustavo", "Henrietta",
    "Ignatius", "Jocelyn", "Kip", "Lavinia", "Mordecai", "Nadia", "Osric",
    "Petra", "Quentin", "Ramona", "Silas", "Tallulah", "Uriel", "Vivienne",
]
LAST_NAMES = [
    "Ashford", "Barrows", "Calloway", "Delacroix", "Ellingson", "Fairbanks",
    "Granger", "Holloway", "Ibarra", "Jessup", "Kowalczyk", "Lindqvist",
    "Marchetti", "Northcutt", "Okafor", "Pemberton", "Quintanilla", "Rutledge",
    "Sandoval", "Thackeray", "Umbridge", "Vandermeer", "Whitlock", "Xiong",
    "Yarborough", "Zielinski", "Abernathy", "Bickford", "Crowder", "Dunmore",
    "Eastwood", "Falkner", "Goodwin", "Hargrove", "Ivester", "Jankowski",
    "Kettering", "Lockhart", "Meriwether", "Nakamura", "Oglesby", "Prescott",
    "Quimby", "Ravenscroft", "Stanfield", "Tremblay", "Underhill", "Villanueva",
    "Wexford", "Yancey", "Zablocki", "Ostrander", "Palafox", "Renwick",
    "Solberg", "Tunstall", "Vasquez-Reed", "Winterbourne", "Alcott", "Bramble",
]

HANDLE_SUFFIXES = ["ForCongress", "4Congress", "ForSenate", "4Senate", "2022",
                   "Campaign", "HQ", "Official"]

def make_name(used):
    while True:
        n = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if n not in used:
            used.add(n)
            return n

def make_handle(name, chamber, used):
    last = name.split()[-1].replace("-", "")
    first = name.split()[0]
    pools = [
        last + rng.choice(HANDLE_SUFFIXES[:2] if chamber == "House" else HANDLE_SUFFIXES[2:4]),
        first + last + "2022",
        last + "HQ",
        first[0] + last + rng.choice(["", "2022", "Official"]),
    ]
    rng.shuffle(pools)
    for h in pools:
        if h.lower() not in used:
            used.add(h.lower())
            return "@" + h
    h = last + str(rng.randint(100, 999))
    used.add(h.lower())
    return "@" + h

def draw_ideology(party):
    if party == "D":
        x = rng.gauss(-0.62, 0.22)
    elif party == "R":
        x = rng.gauss(0.62, 0.22)
    else:
        x = rng.gauss(0.0, 0.45)
    return max(-1.0, min(1.0, x))

TOPICS = ["economy", "healthcare", "immigration", "climate_energy", "guns",
          "abortion", "education", "foreign_policy", "crime_policing",
          "democracy_rights", "campaign_logistics"]

def draw_topic_mix(ideology):
    # Base Dirichlet-ish mix; slight ideological tilt in topic emphasis
    alphas = {
        "economy": 2.2, "healthcare": 1.6, "immigration": 1.3,
        "climate_energy": 1.2, "guns": 0.9, "abortion": 0.9, "education": 1.1,
        "foreign_policy": 0.9, "crime_policing": 1.2, "democracy_rights": 1.0,
        "campaign_logistics": 2.6,
    }
    # Liberals emphasize climate/healthcare/democracy; conservatives immigration/crime/economy
    tilt = ideology
    alphas["climate_energy"] *= (1 - 0.5 * tilt)
    alphas["healthcare"] *= (1 - 0.3 * tilt)
    alphas["democracy_rights"] *= (1 - 0.3 * tilt)
    alphas["immigration"] *= (1 + 0.6 * tilt)
    alphas["crime_policing"] *= (1 + 0.4 * tilt)
    alphas["economy"] *= (1 + 0.15 * tilt)
    draws = {t: rng.gammavariate(a, 1.0) for t, a in alphas.items()}
    s = sum(draws.values())
    return {t: v / s for t, v in draws.items()}


def build_roster():
    used_names, used_handles = set(), set()
    candidates = []
    cid = 0

    def add(chamber, state, district, party, incumbent):
        nonlocal cid
        cid += 1
        name = make_name(used_names)
        ideology = draw_ideology(party)
        candidates.append({
            "candidate_id": f"C{cid:04d}",
            "candidate_name": name,
            "handle": make_handle(name, chamber, used_handles),
            "party": party,
            "chamber": chamber,
            "state": state,
            "district": district,
            "incumbent": incumbent,
            "true_ideology": round(ideology, 4),
            "topic_mix": draw_topic_mix(ideology),
        })

    # House: every district gets a D and an R ~88% of the time each (some
    # uncontested), plus occasional independents.
    for state, n_districts in STATES.items():
        for d in range(1, n_districts + 1):
            dist = f"{state}-{d:02d}" if n_districts > 1 else f"{state}-AL"
            incumbent_party = rng.choice(["D", "R"])
            for party in ("D", "R"):
                contested = rng.random() < (0.97 if party == incumbent_party else 0.88)
                if contested:
                    add("House", state, dist, party,
                        incumbent=(party == incumbent_party and rng.random() < 0.82))
            if rng.random() < 0.045:
                add("House", state, dist, "I", incumbent=False)

    # Senate: 2022 class, both major parties contest everything
    for state in SENATE_2022:
        incumbent_party = rng.choice(["D", "R"])
        for party in ("D", "R"):
            add("Senate", state, f"{state}-Sen", party,
                incumbent=(party == incumbent_party and rng.random() < 0.7))
        if rng.random() < 0.09:
            add("Senate", state, f"{state}-Sen", "I", incumbent=False)

    return candidates

# ----------------------------------------------------------------------------
# Phrase banks: topic x (liberal / neutral / conservative) framing
# ----------------------------------------------------------------------------

PHRASES = {
    "economy": {
        "lib": [
            "Working families deserve a raise. I'll fight to lift the minimum wage and rein in corporate price gouging.",
            "Billionaires shouldn't pay a lower tax rate than teachers. Time to make the wealthy pay their fair share.",
            "Union jobs built the middle class. I'm proud to stand with workers on the picket line in {state}.",
            "Childcare costs are crushing parents in {state}. My plan caps costs and invests in providers.",
            "Corporate consolidation is squeezing family farms and small businesses. We need real antitrust enforcement.",
        ],
        "con": [
            "Small businesses in {state} are drowning in red tape. I'll cut job-killing regulations on day one.",
            "Inflation is a tax on every family. Washington's reckless spending has to stop.",
            "Lower taxes, less government, more freedom. That's how we get {state}'s economy roaring again.",
            "Government doesn't create prosperity — entrepreneurs do. Let's get Washington out of the way.",
            "We must balance the budget like every family in {state} has to. No more blank checks.",
        ],
        "neu": [
            "Met with local business owners in {city} today about what they need to grow and hire.",
            "Good jobs and a fair shot — that's what this campaign is about for {state}.",
            "Talked supply chains and manufacturing with workers at the plant in {city} this morning.",
        ],
    },
    "healthcare": {
        "lib": [
            "Healthcare is a human right. No one in {state} should go bankrupt because they got sick.",
            "I'll fight to expand Medicaid and cap insulin costs for every patient, not just some.",
            "Big Pharma spends millions lobbying to keep prices high. I don't take their money — and I'll vote to negotiate prices.",
            "Rural hospitals in {state} are closing while insurers post record profits. That ends when we hold them accountable.",
        ],
        "con": [
            "Patients and doctors should make health decisions — not government bureaucrats.",
            "Washington-run healthcare means longer waits and fewer choices. I'll defend private options for {state} families.",
            "Price transparency and competition will lower costs faster than any government mandate.",
            "We can protect people with pre-existing conditions without a big-government takeover of medicine.",
        ],
        "neu": [
            "Visited the community health center in {city} — grateful for the nurses and staff who keep it running.",
            "Every family in {state} deserves access to quality, affordable care. Full stop.",
        ],
    },
    "immigration": {
        "lib": [
            "Dreamers are Americans in every way but paperwork. I'll vote for a pathway to citizenship.",
            "Family separation is a moral stain. We need humane, orderly immigration reform — not cruelty as policy.",
            "Immigrants power {state}'s farms, hospitals, and small businesses. Our economy depends on fixing this broken system.",
        ],
        "con": [
            "Secure the border. Period. I'll fund the wall and back our Border Patrol agents 100%.",
            "Sanctuary policies put {state} communities at risk. I'll end them.",
            "Illegal immigration drives down wages for {state} workers. Enforce the law, protect American jobs.",
            "Catch-and-release is over when I get to Washington. We need real deterrence at the border.",
        ],
        "neu": [
            "Our immigration system has been broken for decades. Both parties share the blame — {state} deserves solutions.",
        ],
    },
    "climate_energy": {
        "lib": [
            "The climate crisis is here — {state} families are paying for it in floods, fires, and insurance bills. We need clean energy now.",
            "Clean energy jobs are the future, and {state} can lead. I'll fight for wind, solar, and battery manufacturing here at home.",
            "Big Oil made record profits while your utility bill doubled. I'll end their subsidies.",
            "Protecting our air and water isn't radical — poisoning them for profit is.",
        ],
        "con": [
            "American energy independence means drilling here, mining here, and building pipelines here.",
            "The radical Green agenda would crush {state} jobs and spike your electric bill. I'll stop it.",
            "We can be good stewards of the land without strangling {state}'s energy workers with mandates.",
            "Unleash American energy. Lower prices at the pump start with producing more, not begging OPEC.",
        ],
        "neu": [
            "Toured the new energy facility in {city} today. {state} workers are second to none.",
        ],
    },
    "guns": {
        "lib": [
            "Universal background checks are supported by most gun owners. Congress's inaction is a choice — I'll make a different one.",
            "Our kids practice lockdown drills while Congress practices excuses. Enough. Pass red flag laws.",
            "I respect the Second Amendment AND believe weapons of war don't belong on our streets.",
        ],
        "con": [
            "The Second Amendment isn't negotiable. I'll oppose every gun grab, every time.",
            "Criminals don't follow gun laws — disarming law-abiding {state} citizens makes us less safe.",
            "I'm a proud gun owner and I'll defend your constitutional rights in Washington.",
        ],
        "neu": [
            "Met with law enforcement and community leaders in {city} about keeping our neighborhoods safe.",
        ],
    },
    "abortion": {
        "lib": [
            "Abortion is healthcare, and healthcare decisions belong to patients — not politicians.",
            "I will vote to restore Roe's protections nationwide. {state} women deserve nothing less.",
            "Politicians have no place in the exam room. I trust women.",
        ],
        "con": [
            "I am proudly pro-life and will always defend the unborn.",
            "Life is precious. I'll support policies that protect mothers and babies alike.",
            "Taxpayer dollars should never fund abortion. I'll hold that line in Congress.",
        ],
        "neu": [
            "This issue is deeply personal for families across {state}. I'll always listen first.",
        ],
    },
    "education": {
        "lib": [
            "Every kid in {state} deserves a great public school — fully funded, with teachers paid what they're worth.",
            "Student debt is holding back a generation. I support relief and free community college.",
            "Book bans don't protect kids. Librarians and teachers deserve our trust, not harassment.",
        ],
        "con": [
            "Parents — not bureaucrats — should decide what their kids learn. I'll fight for parental rights in every classroom.",
            "School choice gives every {state} family a shot at a great education, regardless of zip code.",
            "Get politics out of the classroom and get back to reading, writing, and math.",
        ],
        "neu": [
            "Stopped by {city} Elementary today. Our teachers are heroes — period.",
        ],
    },
    "foreign_policy": {
        "lib": [
            "Diplomacy first. Endless wars cost {state} families dearly — in lives and in dollars.",
            "Standing with our allies isn't weakness, it's strength. Alliances keep Americans safe.",
            "Human rights must be at the center of American foreign policy, not an afterthought.",
        ],
        "con": [
            "Peace through strength. A strong military is the best guarantee our enemies stay in check.",
            "China is eating our lunch on trade and technology. I'll put American workers first.",
            "Not one more dime overseas until we take care of our own veterans here in {state}.",
        ],
        "neu": [
            "Honored to meet with veterans in {city} today. We owe them more than words.",
        ],
    },
    "crime_policing": {
        "lib": [
            "Real public safety means investing in communities — mental health response, youth programs, and accountable policing.",
            "We can back the police AND demand accountability. Those aren't opposites.",
            "Gun violence is a public health crisis. Prevention works — I've seen it in {city}.",
        ],
        "con": [
            "I'll always back the blue. Our officers in {state} deserve support, not defunding.",
            "Soft-on-crime policies have consequences. I'll fight for tougher sentences for violent offenders.",
            "Every family deserves a safe neighborhood. Law and order is on the ballot.",
        ],
        "neu": [
            "Rode along with {city} officers last night. Grateful for what they do every shift.",
        ],
    },
    "democracy_rights": {
        "lib": [
            "Voting should be easy and secure — for everyone. I'll fight voter suppression wherever it appears.",
            "Dark money is drowning out {state} voices. I support full disclosure of every political dollar.",
            "Gerrymandering lets politicians pick their voters. Independent commissions would fix that.",
        ],
        "con": [
            "Election integrity matters. Voter ID is common sense and most Americans agree.",
            "The Constitution isn't a suggestion. I'll defend free speech and religious liberty against any government overreach.",
            "Washington elites think they know better than {state} voters. I trust the people.",
        ],
        "neu": [
            "Democracy works when we show up. Check your registration today — link in bio.",
        ],
    },
    "campaign_logistics": {
        "lib": [], "con": [],
        "neu": [
            "Join us Saturday at {time} for a town hall in {city}! Doors open early — bring your questions.",
            "We just crossed {num} grassroots donors! Average gift: $27. This campaign is powered by people, not PACs.",
            "Knocking doors in {city} with an incredible team of volunteers today. The energy is real!",
            "Thank you {city}! Standing room only tonight. This movement is growing every single day.",
            "New endorsement dropping tomorrow morning. Stay tuned, {state}!",
            "Debate night. Watch live at {time} and see the contrast for yourself.",
            "Early voting starts soon in {state}. Make your plan now — every single vote counts.",
            "Phone bank tonight at {time}! Grab a coffee and join us — training provided, no experience needed.",
            "Our new ad is live. Watch it, share it, and chip in $5 if you can.",
            "Yard signs are IN. Swing by the {city} office and grab one this weekend!",
            "Filed our FEC report: best fundraising quarter yet. Thank you to every supporter.",
            "Happy {holiday} from our family to yours, {state}!",
        ],
    },
}

CITIES = ["Riverton", "Cedar Falls", "Millbrook", "Oak Ridge", "Fairview",
          "Lakewood", "Springdale", "Bremerton", "Kingsport", "Alton",
          "Greenville", "Madison Heights", "Clayton", "Bellview", "Harrisburg",
          "Newport", "Dover Plains", "Summit", "Georgetown", "Franklin"]
HOLIDAYS = ["Fourth of July", "Memorial Day", "Labor Day", "Thanksgiving",
            "Veterans Day", "New Year"]

def fill(template, state):
    return (template
            .replace("{state}", state)
            .replace("{city}", rng.choice(CITIES))
            .replace("{time}", rng.choice(["10am", "noon", "2pm", "5:30pm", "6pm", "7pm"]))
            .replace("{num}", f"{rng.choice([5, 10, 15, 20, 25, 50])},000")
            .replace("{holiday}", rng.choice(HOLIDAYS)))

# ----------------------------------------------------------------------------
# Fictional retweet-source accounts (orgs/pundits) with their own ideology
# ----------------------------------------------------------------------------

RT_SOURCES = []
_rt_specs = [
    # (handle, ideology)
    ("@ProgressNowDaily", -0.85), ("@GreenFutureFund", -0.8),
    ("@WorkersUnitedHQ", -0.75), ("@CarePolicyProject", -0.7),
    ("@VotingRightsWatch", -0.65), ("@CleanAirCoalition", -0.6),
    ("@TheDailyLedgerUS", -0.45), ("@PublicSquareNews", -0.3),
    ("@HeartlandDispatch", -0.15), ("@CivicSignal", 0.0),
    ("@StatehouseWire", 0.1), ("@MainStreetMonitor", 0.2),
    ("@LibertyLedger", 0.4), ("@FreeMarketForum", 0.55),
    ("@BorderWatchNet", 0.65), ("@FaithAndFlagDaily", 0.7),
    ("@SecondAmendmentNow", 0.75), ("@TaxpayersFirstOrg", 0.8),
    ("@PatriotPressWire", 0.85), ("@HeritageHorizon", 0.9),
]
RT_TEXTS = {
    "lib": [
        "NEW REPORT: Corporate profits hit record highs while wages stagnate. The data is undeniable.",
        "BREAKING: Another state expands Medicaid — and rural hospitals immediately benefit.",
        "This map of clean-energy job growth should be front-page news.",
        "Study: universal pre-K pays for itself within a decade. Pass it on.",
        "Voter purges are quietly accelerating ahead of the midterms. We're tracking every state.",
    ],
    "con": [
        "REPORT: Federal regulations cost small businesses billions last year. Time to cut the red tape.",
        "Border crossings surge again — where is the accountability?",
        "New analysis: energy mandates will raise household electric bills sharply. Voters deserve the truth.",
        "Crime stats out this morning. Soft-on-crime policies have a body of evidence now — and it's damning.",
        "Poll: overwhelming majority supports voter ID. This shouldn't be controversial.",
    ],
    "neu": [
        "Our latest polling roundup for the 2022 midterms is live. Key races tightening.",
        "ICYMI: our candidate tracker now covers every House and Senate race. Bookmark it.",
        "Debate schedule for the fall is out. Full calendar at the link.",
        "Turnout in this week's primaries beat 2018 levels in most counties.",
    ],
}
for h, ideo in _rt_specs:
    RT_SOURCES.append({"handle": h, "ideology": ideo})

# ----------------------------------------------------------------------------
# Tweet generation
# ----------------------------------------------------------------------------

CAMPAIGN_START = datetime(2021, 9, 1)
CAMPAIGN_END = datetime(2022, 11, 8)  # through Election Day 2022
SPAN_DAYS = (CAMPAIGN_END - CAMPAIGN_START).days

HOUR_WEIGHTS = [0.2, 0.1, 0.05, 0.05, 0.05, 0.1, 0.5, 1.5, 2.5, 3.0, 3.0, 2.8,
                2.6, 2.5, 2.4, 2.3, 2.5, 2.8, 3.0, 3.2, 2.8, 2.0, 1.2, 0.5]

def lean_probs(ideology):
    """P(lib), P(neu), P(con) as a function of latent ideology."""
    p_con = 1.0 / (1.0 + math.exp(-3.2 * ideology))
    p_lib = 1.0 - p_con
    # squeeze toward neutral a bit; moderates tweet neutral more often
    p_neu = 0.28 + 0.22 * (1 - abs(ideology))
    scale = 1 - p_neu
    return p_lib * scale, p_neu, p_con * scale

def draw_timestamp():
    # Ramp: activity grows over campaign; quadratic ramp
    u = rng.random() ** 0.55  # skews toward 1 -> later dates more likely
    day = int(u * SPAN_DAYS)
    hour = rng.choices(range(24), weights=HOUR_WEIGHTS)[0]
    return CAMPAIGN_START + timedelta(days=day, hours=hour,
                                      minutes=rng.randint(0, 59),
                                      seconds=rng.randint(0, 59))

def pick_lean(ideology):
    pl, pn, pc = lean_probs(ideology)
    return rng.choices(["lib", "neu", "con"], weights=[pl, pn, pc])[0]

def gen_original_tweet(cand):
    mix = cand["topic_mix"]
    topic = rng.choices(list(mix.keys()), weights=list(mix.values()))[0]
    bank = PHRASES[topic]
    if topic == "campaign_logistics":
        lean = "neu"
    else:
        lean = pick_lean(cand["true_ideology"])
        if not bank[lean]:
            lean = "neu"
    text = fill(rng.choice(bank[lean]), cand["state"])
    return topic, lean, text

def gen_retweet(cand):
    # candidates retweet sources near their own ideology
    weights = [math.exp(-((s["ideology"] - cand["true_ideology"]) ** 2) / (2 * 0.28 ** 2)) + 0.02
               for s in RT_SOURCES]
    src = rng.choices(RT_SOURCES, weights=weights)[0]
    lean = pick_lean(src["ideology"])
    text = rng.choice(RT_TEXTS[lean])
    # NOTE (documented limitation, do not "fix" — the shipped corpus is
    # hash-pinned in ws0-harness): `lean` is drawn here so retweet text IS
    # framing-coded, but the caller records true_framing="" for retweets.
    # The planted framing truth for the 26.5% retweet rows is therefore
    # discarded, and framing recovery cannot be evaluated on retweets.
    return src["handle"], f"RT {src['handle']}: {text}"

def main():
    candidates = build_roster()
    print(f"Roster: {len(candidates)} candidates")

    rows = []
    tweet_serial = 0
    for cand in candidates:
        # activity level: lognormal, mean ~115
        n_tweets = max(8, int(rng.lognormvariate(math.log(100), 0.55)))
        # retweet share varies per candidate: Beta(2.2, 6) -> mean ~0.27
        rt_share = rng.betavariate(2.2, 6.0)
        for _ in range(n_tweets):
            tweet_serial += 1
            ts = draw_timestamp()
            is_rt = rng.random() < rt_share
            if is_rt:
                src_handle, text = gen_retweet(cand)
                topic, lean = "retweet_source", ""
            else:
                topic, lean, text = gen_original_tweet(cand)
                src_handle = ""
            rows.append({
                "tweet_id": f"T{tweet_serial:07d}",
                "candidate_id": cand["candidate_id"],
                "candidate_name": cand["candidate_name"],
                "handle": cand["handle"],
                "party": cand["party"],
                "chamber": cand["chamber"],
                "state": cand["state"],
                "district": cand["district"],
                "incumbent": cand["incumbent"],
                "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "is_retweet": is_rt,
                "retweeted_handle": src_handle,
                "text": text,
                "true_topic": topic,
                "true_framing": lean,
                "true_ideology": cand["true_ideology"],
            })

    # global chronological order (firehose-style)
    rows.sort(key=lambda r: r["timestamp_utc"])
    # re-issue tweet IDs in chronological order so IDs sort like real snowflake IDs
    for i, r in enumerate(rows, 1):
        r["tweet_id"] = f"T{i:07d}"

    # anchor output next to this script (not the cwd), and don't leave an
    # uncompressed ~30MB duplicate behind
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "synthetic_candidate_tweets_2022.csv")
    fields = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # mtime=0 so the .gz bytes are identical across reruns (gzip's default
    # header embeds the current time, so byte-identical CSV content would
    # otherwise hash differently every run). NOTE: the 2026-07 committed
    # artifact was written without mtime=0, so even a faithful regeneration
    # will not byte-match the hash pinned in ws0-harness — the CSV *content*
    # is identical (seed-determined), only the gzip header differs. If you
    # regenerate by accident, restore the pinned artifact with
    # `git checkout -- :/data/synthetic-candidate-tweets/` (the :/ prefix
    # makes the pathspec repo-rooted, so it works from any directory).
    with open(out, "rb") as f_in, \
            gzip.GzipFile(out + ".gz", "wb", mtime=0) as f_out:
        f_out.writelines(f_in)
    os.remove(out)

    print(f"Tweets: {len(rows)} rows -> {out}.gz")

if __name__ == "__main__":
    main()
