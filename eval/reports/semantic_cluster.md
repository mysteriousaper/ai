# Engine F (semantic cluster map v0.1: MiniLM + UMAP + HDBSCAN) evaluation

## GoEmotions test (n=5427), single global threshold 0.09 tuned on validation, 95% CI (200 resamples)

| Model | Macro F1 | Ekman-6 F1 (first 2000, thr 0.3) |
|---|---|---|
| Semantic cluster (F) | 0.126 [0.120, 0.133] | 0.069 |
| Transformer (B), per-label tuned thresholds | 0.522 [0.499, 0.540] | 0.628 |
| Lexicon (A) | n/a (see lexicon.md) | 0.245 [0.229, 0.259] |
| Majority ('neutral') | 0.018 | 0.093 |

## Cluster quality (train corpus, from build log / metrics.json)

| Metric | HDBSCAN | K-means (same k) |
|---|---|---|
| n leaf clusters | 75 | 75 |
| noise share | 0.743 | 0 |
| NMI vs primary label | 0.120 (non-noise), 0.050 (noise as cluster) | 0.064 (all), 0.110 (same pts) |
| purity (majority primary share 0.295) | 0.355 | 0.328 |
| stability ARI, 2 other seeds/samples | 0.938, 0.950 | n/a |
| min_cluster_size (+ selection; grid 50/100/200 eom, 50/100 leaf) | 50 (leaf) | |

## Top-level boxes (8) and child leaves

### Box 0: neutral/love | love, oh, thank (n=852)
- leaf 64 (n=206): neutral 0.53, admiration 0.07, love 0.07; words: she's, girl, woman, broke, poor
- leaf 72 (n=204): love 0.53, admiration 0.21, neutral 0.12; words: love, loved, loves, names, favorite
- leaf 73 (n=152): neutral 0.46, anger 0.08, surprise 0.07; words: oh, damn, bye, beast, uh
- leaf 71 (n=123): neutral 0.48, annoyance 0.09, approval 0.09; words: limits, cares, know, got, problem
- leaf 74 (n=86): gratitude 0.67, neutral 0.08, admiration 0.07; words: thank, bless, thanks, dear, happyfriendlybot
- leaf 57 (n=81): neutral 0.48, curiosity 0.11, approval 0.07; words: supporters, vote, voted, stop, dumbass

### Box 1: neutral/annoyance | religion, women, racist (n=1651)
- leaf 52 (n=379): neutral 0.27, annoyance 0.25, disapproval 0.11; words: opinion, argument, stupid, ignorant, point
- leaf 37 (n=288): neutral 0.41, approval 0.10, disapproval 0.10; words: women, men, gender, woman, feminism
- leaf 29 (n=187): neutral 0.33, curiosity 0.11, approval 0.09; words: religion, church, atheist, religious, atheists
- leaf 46 (n=169): neutral 0.52, annoyance 0.07, confusion 0.07; words: left, conservative, referendum, liberals, brexit
- leaf 54 (n=130): neutral 0.36, disapproval 0.09, amusement 0.07; words: sub, repost, subreddit, posted, post
- leaf 45 (n=121): neutral 0.46, disapproval 0.12, amusement 0.07; words: communism, capitalism, socialism, anarchist, socialist
- leaf 39 (n=104): neutral 0.39, annoyance 0.14, disapproval 0.10; words: racist, racism, white, racists, race
- leaf 10 (n=75): neutral 0.49, disapproval 0.09, annoyance 0.08; words: news, buzzfeed, media, journalists, fox
- leaf 28 (n=75): neutral 0.32, annoyance 0.20, anger 0.12; words: care, people, person, respect, act
- leaf 53 (n=66): neutral 0.46, annoyance 0.15, disapproval 0.08; words: removed, rule, rules, posts, comments
- leaf 34 (n=57): anger 0.40, neutral 0.26, annoyance 0.19; words: angry, anger, mad, frustration, calm

### Box 2: neutral/admiration | team, game, season (n=1288)
- leaf 70 (n=533): neutral 0.40, admiration 0.12, optimism 0.07; words: player, team, game, he's, season
- leaf 68 (n=475): neutral 0.29, disappointment 0.10, admiration 0.08; words: team, win, game, fans, lose
- leaf 69 (n=280): neutral 0.39, admiration 0.10, disapproval 0.09; words: draft, pick, trade, season, team

### Box 3: neutral/admiration | video, joke, cat (n=1651)
- leaf 27 (n=211): neutral 0.35, annoyance 0.08, approval 0.08; words: she's, girlfriend, life, tell, want
- leaf 20 (n=186): neutral 0.33, love 0.16, admiration 0.15; words: song, music, songs, album, listen
- leaf 15 (n=128): neutral 0.45, surprise 0.09, curiosity 0.06; words: head, hit, dude, helmet, accident
- leaf 24 (n=128): admiration 0.23, neutral 0.19, love 0.18; words: episode, season, episodes, series, watched
- leaf 26 (n=116): admiration 0.47, neutral 0.18, love 0.13; words: looks, she's, looked, cute, pic
- leaf 42 (n=96): sadness 0.39, neutral 0.21, joy 0.09; words: crying, tears, eyes, joy, i'm
- leaf 23 (n=95): admiration 0.32, neutral 0.24, love 0.10; words: movie, movies, film, netflix, watch
- leaf 47 (n=90): amusement 0.73, gratitude 0.17, joy 0.12; words: funny, laughed, laugh, chuckle, loud
- leaf 2 (n=84): neutral 0.48, annoyance 0.10, anger 0.06; words: personal, police, space, boring, target
- leaf 16 (n=84): admiration 0.34, love 0.30, amusement 0.12; words: he's, interview, funny, love, awesome
- leaf 17 (n=82): neutral 0.28, admiration 0.23, curiosity 0.12; words: video, videos, youtube, vid, footage
- leaf 21 (n=78): neutral 0.39, admiration 0.10, love 0.08; words: cat, cats, kitty, kittens, seen
- leaf 5 (n=77): neutral 0.34, admiration 0.13, love 0.09; words: art, poster, painting, artist, draw
- leaf 22 (n=72): neutral 0.29, admiration 0.14, love 0.10; words: dog, dogs, pet, trained, pup
- leaf 0 (n=66): neutral 0.26, fear 0.21, annoyance 0.12; words: cringe, cringey, cringy, what's, lmao
- leaf 48 (n=58): amusement 0.38, neutral 0.21, confusion 0.09; words: joke, funny, jokes, explain, hilarious

### Box 4: admiration/gratitude | happy, cake, birthday (n=747)
- leaf 67 (n=204): gratitude 0.69, admiration 0.28, neutral 0.03; words: thank, thanks, awesome, appreciate, welcome
- leaf 60 (n=192): admiration 0.55, love 0.33, excitement 0.06; words: amazing, love, awesome, beautiful, brilliant
- leaf 41 (n=84): love 0.89, admiration 0.08, approval 0.02; words: love, username, homo, like, say
- leaf 61 (n=82): excitement 0.29, joy 0.27, gratitude 0.23; words: new, year, happy, cheers, years
- leaf 38 (n=70): joy 0.43, excitement 0.39, gratitude 0.09; words: cake, day, happy, cakeday, icing
- leaf 51 (n=62): excitement 0.32, joy 0.32, neutral 0.18; words: birthday, happy, day, hope, tomorrow
- leaf 63 (n=53): joy 0.43, gratitude 0.36, admiration 0.30; words: happy, congrats, congratulations, positive, hope

### Box 5: neutral/gratitude | game, thanks, play (n=2089)
- leaf 12 (n=728): neutral 0.32, admiration 0.10, approval 0.09; words: game, play, games, guns, gun
- leaf 58 (n=309): neutral 0.34, approval 0.21, amusement 0.20; words: true, oh, lol, damn, yes
- leaf 65 (n=209): gratitude 0.72, remorse 0.09, admiration 0.08; words: thanks, ah, thank, clarification, sense
- leaf 6 (n=194): neutral 0.34, annoyance 0.08, confusion 0.08; words: phone, mobile, phones, app, problem
- leaf 11 (n=160): neutral 0.29, approval 0.10, curiosity 0.09; words: old, age, older, years, olds
- leaf 50 (n=136): neutral 0.49, approval 0.09, disapproval 0.07; words: money, tax, pay, income, taxes
- leaf 4 (n=106): neutral 0.42, curiosity 0.13, optimism 0.07; words: late, time, days, hours, wait
- leaf 66 (n=66): gratitude 0.77, joy 0.14, optimism 0.08; words: helpful, thank, help, thanks, glad
- leaf 49 (n=64): neutral 0.42, curiosity 0.08, approval 0.06; words: free, price, market, cash, cheap
- leaf 1 (n=63): neutral 0.32, curiosity 0.14, approval 0.11; words: number, numbers, zero, statistic, odds
- leaf 3 (n=54): neutral 0.56, surprise 0.07, realization 0.06; words: years, ago, future, old, decade

### Box 6: neutral/caring | kids, mom, luck (n=1117)
- leaf 59 (n=190): optimism 0.33, caring 0.27, remorse 0.12; words: hope, better, sorry, strong, stay
- leaf 18 (n=161): neutral 0.37, approval 0.08, annoyance 0.07; words: kids, children, child, parents, parent
- leaf 35 (n=157): neutral 0.38, caring 0.19, annoyance 0.08; words: ask, he's, sounds, tell, leave
- leaf 19 (n=152): neutral 0.22, admiration 0.16, gratitude 0.10; words: mom, dad, mother, mum, daddy
- leaf 55 (n=104): neutral 0.34, admiration 0.14, anger 0.09; words: vaccines, cancer, doctor, vaccinate, measles
- leaf 56 (n=74): neutral 0.32, caring 0.16, gratitude 0.11; words: therapist, therapy, anxiety, mental, health
- leaf 43 (n=73): neutral 0.23, approval 0.11, annoyance 0.10; words: school, teacher, class, students, college
- leaf 36 (n=70): neutral 0.50, approval 0.13, curiosity 0.09; words: friends, friend, group, make, people
- leaf 44 (n=70): neutral 0.30, amusement 0.09, optimism 0.09; words: sleep, wake, bed, morning, waking
- leaf 62 (n=66): optimism 0.39, caring 0.26, gratitude 0.21; words: luck, good, hoping, friend, best

### Box 7: neutral/admiration | pizza, hair, food (n=1757)
- leaf 31 (n=610): neutral 0.35, admiration 0.16, love 0.10; words: pizza, food, eat, chicken, cheese
- leaf 7 (n=197): neutral 0.36, annoyance 0.11, disapproval 0.08; words: car, traffic, drive, road, driver
- leaf 9 (n=161): admiration 0.24, neutral 0.22, love 0.09; words: live, city, area, florida, moved
- leaf 14 (n=160): neutral 0.41, admiration 0.14, approval 0.08; words: hair, eyes, eye, eyebrows, shave
- leaf 40 (n=149): neutral 0.39, annoyance 0.11, disapproval 0.11; words: police, cops, jail, arrest, fbi
- leaf 13 (n=127): neutral 0.33, love 0.10, approval 0.09; words: shoes, shirt, wear, wearing, hat
- leaf 33 (n=107): neutral 0.33, approval 0.14, anger 0.07; words: weed, drugs, smoking, drug, marijuana
- leaf 32 (n=71): neutral 0.39, approval 0.11, desire 0.07; words: beer, drink, drunk, drinking, beers
- leaf 30 (n=68): neutral 0.41, disgust 0.13, disapproval 0.07; words: smell, smells, toothpaste, smelly, teeth
- leaf 8 (n=56): neutral 0.41, disapproval 0.11, admiration 0.09; words: snow, cold, weather, winter, rain
- leaf 25 (n=51): neutral 0.41, admiration 0.08, disappointment 0.08; words: fat, weight, overweight, underweight, exercise

## Verdict

- As an emotion classifier F is weak (macro F1 0.126 vs B 0.522): it mixes neighbour label distributions, so rare labels never cross one global threshold (Ekman-6 at the fixed 0.3 threshold is even below the majority baseline for the same reason). Known failure: 'this is terrifying' lands in an admiration leaf (xfail test). It was not meant to beat B.
- Its value is neighbourhoods and explanation: nearest leaf, box path, words and example texts. Clusters are semantic/topical more than purely emotional (NMI vs label 0.12); stability ARI 0.94.
- English-only (MiniLM-L6). Leaf assignment uses nearest 384-D prototype, not HDBSCAN prediction, so it can differ from the offline labels. Do not read boxes as natural psychological categories.
