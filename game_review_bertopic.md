# Game Review BERTopic

- Generated: 2026-06-23T16:15:02
- Input: `C:\Users\admin\Documents\studable query latent\game_review_data\game_review_cleaned_3_sentences`
- Input files: 293
- Input directory size: 99.97 GB
- Sample method: `balanced_prefix_per_game`
- Sampled documents: 100000
- Embedding dimension: 1024
- HDBSCAN `min_cluster_size`: 100
- Random state: 42
- Fit time: 84.1 seconds

Note: the source sentence/vector directory is about 100 GB. This run uses a deterministic per-game prefix sample and skips metadata records with `review_id < 3` before fitting BERTopic. It is a practical topic snapshot, not a full all-sentence HDBSCAN fit.

## Environment

| Package | Version |
|---|---:|
| `bertopic` | `0.17.4` |
| `hdbscan` | `0.8.44` |
| `umap-learn` | `0.5.12` |
| `ijson` | `3.5.0` |
| `numpy` | `2.4.3` |
| `scikit-learn` | `1.8.0` |

## Summary

- Topics excluding outliers: 70
- Outlier documents: 57079
- Outlier rate: 57.08%

## Topic Table

| Topic | Count | Top Words |
|---:|---:|---|
| -1 | 57079 | outliers |
| 0 | 15674 | boss, npc, bug, ai, ui, rpg, jrpg, buff, cd, ps |
| 1 | 2815 | hate, thats, just, im, thing, really, say, good, dont, bad |
| 2 | 1511 | cars, car, racing, f1, race, races, nfs, speed, vehicles, vehicle |
| 3 | 1466 | game, recommend, recommend game, review, love game, love, games, game game, play, fun |
| 4 | 1264 | et, le, les, jeu, des, pas, qui, vous, pour, je |
| 5 | 1087 | hours, horas, hours game, stunden, hour, minutes, took, ive, time, played |
| 6 | 918 | music, soundtrack, sound, sounds, sound design, songs, audio, design, tracks, song |
| 7 | 883 | dark souls, dead cells, cells, souls, rip, skyrim, dark, elden ring, elden, west |
| 8 | 839 | fps, settings, pc, performance, cpu, high, frame, stuttering, running, frames |
| 9 | 765 | weapons, weapon, guns, gun, ammo, use, shotgun, armas, different, waffen |
| 10 | 685 | steam, ea, origin, launcher, launch, files, account, install, game steam, download |
| 11 | 654 | sale, price, worth, buy, preo, wait, money, precio, cheaper, worth price |
| 12 | 651 | xcom, souls, rpg, soulslike, dark souls, dark, gears, like, games, roguelike |
| 13 | 562 | building, farming, buildings, build, farm, base, crops, houses, und, trees |
| 14 | 522 | dlc, dlcs, pack, edition, price, content, deluxe, packs, buy, base game |
| 15 | 500 | map, maps, mapa, mapas, die, markers, und, world, way, open |
| 16 | 491 | ror, goh, ps2, heat, dirt, nier automata, automata, melty, ryu, nier |
| 17 | 481 | combat, fun, fighting, battle, fights, fight, mechanics, really, game combat, battles |
| 18 | 478 | graphics, art, animations, models, visuals, grafik, style, beautiful, grficos, pixel |
| 19 | 468 | price, sale, worth, game, recommend, preo, jogo, buy, game worth, price tag |
| 20 | 367 | dlc, jrpg, ost, rpg, katana, fortune, deaths, bgm, ai, ex |
| 21 | 355 | story, plot, stories, good story, interesting, written, good, predictable, ending, storyline |
| 22 | 352 | review, reviews, negative, write, comments, positive, read, helpful, change review, im |
| 23 | 332 | enemies, enemy, attacks, dodge, attack, units, hit, damage, gegner, opponent |
| 24 | 328 | devs, developers, updates, patch, dev, update, entwickler, feedback, fix, developer |
| 25 | 310 | boss, bosses, fight, boss fights, fights, final, patterns, beat, fun, attack |
| 26 | 307 | una, minutos, la, para, potato, las, hasta, luego, el, en |
| 27 | 304 | controller, mouse, controls, keyboard, control, controllers, xbox, teclado, keys, support |
| 28 | 301 | coop, multiplayer, play, online, friends, mode, players, works, experience, single player |
| 29 | 297 | early access, early, access, game early, release, beta, released, game, content, title |
| 30 | 288 | difficulty, hard, difficulties, normal, easy, hardest, skill, higher, higher difficulties, challenge |
| 31 | 287 | level, skills, skill, perks, gear, upgrade, leveling, points, tree, equipment |
| 32 | 275 | computer, ask, spare, paint, check, press, difficulty, potato, run, decent |
| 33 | 263 | characters, character, personality, cast, memorable, personalities, feel, written, unique, depth |
| 34 | 255 | quests, quest, story, main, main story, complete, multiple, haunting, arent, stories |
| 35 | 254 | sex, love, lust, candy, desktop, cute, queen, party, metal, challenge |
| 36 | 241 | achievements, cosmetics, gold, currency, gems, ingame, items, achievement, buy, premium |
| 37 | 237 | english, translation, language, russian, translate, las, que, los, google, es |
| 38 | 228 | ai, ki, ia, dumb, die, teammates, bad, bots, dlc, bir |
| 39 | 223 | ban, banned, account, hackers, discord, post, forums, people, reddit, han |
| 40 | 220 | voice, voice acting, acting, actors, voiced, voices, lines, english, great, characters |
| 41 | 219 | missions, mission, main, story, misiones, objective, maybe, doing, las misiones, trial |
| 42 | 198 | animals, deer, animal, pet, dog, feed, die, hunting, wenn, bears |
| 43 | 195 | remaster, remake, remastered, original, edition, rereckoning, collection, que, job, juego |
| 44 | 191 | city, climb, environment, location, beautiful, gorgeous, scenery, trees, environments, forest |
| 45 | 190 | persona, p5, pc, golden, jrpgs, que joguei, na sua, vida, ps5, played |
| 46 | 184 | no valid tokens |
| 47 | 184 | servers, server, online, connection, issues, ping, los, players, play, die |
| 48 | 172 | game, missing, mechanics, core, design, ideas, feels, gameplay, new, overall game |
| 49 | 171 | chinese, translation, mod, hard, npc, determine, hard work, bug, simplified, really great |
| 50 | 165 | ships, ship, fleet, salvage, cargo, pulling, apart, shifts, youll, right |
| 51 | 159 | bug, mod, online, tps, auf, bedeutet, der, gefhl, zwei, aus |
| 52 | 151 | save, reload, saving, load, saves, file, manual, manually, quit, checkpoint |
| 53 | 150 | yakuza, dragon, series, like, games, ps2, la, entries, jrpg, youre |
| 54 | 143 | ui, menu, interface, main menu, menus, clunky, confusing, main, horrible, screens |
| 55 | 143 | art, visuals, visually, graphics, art style, visual, style, game, stunning, gorgeous |
| 56 | 140 | recommend, recommended, recommendation, highly, caveats, highly recommend, definitely recommend, im going, current state, current |
| 57 | 139 | pvp, pve, mode, team, server, players, que, solo, competitive, modes |
| 58 | 127 | spieler, ich, competition, tekken, von, der, chain, gwent, habe, den |
| 59 | 115 | deck, cards, decks, card, karten, draw, building, hand, drawing, sts |
| 60 | 109 | matchmaking, lobby, match, lobbies, queue, matches, friends, players, times, playing |
| 61 | 106 | cup, replace, long, itll, game time, average, short, life, price, time |
| 62 | 105 | brain, usage, significant, project, problems, uso, hard master, learn hard, master, easy learn |
| 63 | 105 | survival, stay, decent, beautiful, reality, forget, true, bad, graphics, good |
| 64 | 105 | stardew, valley, stardew valley, crossing, animal, sun, harvest, farming, core, cozy |
| 65 | 105 | chinese, need, grind, ir, dark souls, souls, nos, dark, difficult, van |
| 66 | 104 | camera, view, person, fov, mouse, movement, position, controls, driving, click |
| 67 | 103 | classes, class, different, character, classe, playstyle, differently, play, cards, que |
| 68 | 103 | fishing, fish, catch, minigame, click, simple, sell, minigames, mini, mechanic |
| 69 | 102 | histria, uma histria, ateno, uma, jogabilidade, tem uma, tem, bem, vida, sobre |

## Top Topic Examples

### Topic 0 (15674 docs)

Top words: boss, npc, bug, ai, ui, rpg, jrpg, buff, cd, ps

- `1009290_6672.json review=3 sentence=sentence_2`: 游戏近期更新，这下可以彻底放弃这款粪作了~
- `1009290_6672.json review=3 sentence=sentence_3`: 弃掉之前，做做好事，还是说说原因避免别人被坑吧~
- `1009290_6672.json review=3 sentence=sentence_4`: 游戏近期更新了一个【极】难度的迷宫，里面的BOSS高达破天荒的210级，血量高达1亿多，然而我们角色还是50级。

### Topic 1 (2815 docs)

Top words: hate, thats, just, im, thing, really, say, good, dont, bad

- `1009290_6672.json review=5 sentence=sentence_2`: Far from it actually.
- `1009290_6672.json review=5 sentence=sentence_41`: I guess there is a lot you can do though.
- `1009290_6672.json review=5 sentence=sentence_50`: It is A LOT... way too much to cover here though.

### Topic 2 (1511 docs)

Top words: cars, car, racing, f1, race, races, nfs, speed, vehicles, vehicle

- `1016800_13368.json review=5 sentence=sentence_33`: 然后我又被打死了，再读档，这次虽然油仍旧没有，但是在车前盖上给我刷了一个电子零件！
- `1016950_4196.json review=15 sentence=sentence_2`: J'ai joué plusieurs races, plusieurs compétitions et matches.
- `1029690_19693.json review=9 sentence=sentence_4`: Arabanın içinde bir npc var araca sıkıyorsun öylece içinde oturuyor.

### Topic 3 (1466 docs)

Top words: game, recommend, recommend game, review, love game, love, games, game game, play, fun

- `1012790_10355.json review=3 sentence=sentence_29`: Needless to say, I can highly recommend the game in its current form, hell I could have given it a moderately strong recommendation in the 1.0, but now I can even promise, it will get even better than it is now, and I already love it.
- `1012790_10355.json review=4 sentence=sentence_12`: YOU CAN TRUST THIS GAME TO NOT HAVE YO ♥♥♥♥
- `1012790_10355.json review=5 sentence=sentence_15`: That was when I learned you should never get cocky in this game.

### Topic 4 (1264 docs)

Top words: et, le, les, jeu, des, pas, qui, vous, pour, je

- `1016800_13368.json review=12 sentence=sentence_7`: Une jauge de santé (normale) mais notamment une de santé mentale, car oui lorsque vous tuez un ennemie celle-ci baisse.
- `1016800_13368.json review=12 sentence=sentence_12`: On sent que les développeurs s'investissent pas mal et ça c'est cool : _ _
- `1016950_4196.json review=8 sentence=sentence_1`: Un jeu que j'attendais car en temps que pigeon assumé cela me fait toujours plaisir de dépenser mon argent pour des évolutions des jeux que j'aime.

### Topic 5 (1087 docs)

Top words: hours, horas, hours game, stunden, hour, minutes, took, ive, time, played

- `1009290_6672.json review=4 sentence=sentence_5`: Se você não rushar a historia, você leva no minimo 100 horas pra zerar o jogo igual a mim ;D
- `1009290_6672.json review=5 sentence=sentence_3`: I am halfway through the main campaign with ~50 hours of play-time which is well above average for campaigns nowadays.
- `1009290_6672.json review=5 sentence=sentence_7`: There's actually a lot of content that you could play Co-op, but if you do a 'long-play' through Chapter 1 (not skipping any dialogue and choosing to opt for the 'long-play' option) then expect anywhere from 10-18 hours for Chapter 1 to be completed.

### Topic 6 (918 docs)

Top words: music, soundtrack, sound, sounds, sound design, songs, audio, design, tracks, song

- `1012790_10355.json review=13 sentence=sentence_15`: To add to that stomach churning sea of what the f'k, you then add the sound design.
- `1012790_10355.json review=13 sentence=sentence_17`: To put this freaky make you piss yourself in paranoia type of noises into the game?
- `1012790_10355.json review=13 sentence=sentence_18`: It's very ambient...there's a befouled wind blowing...chairs making creepy classroom noises...something unnatural either creaking, or sucking out a soul, hard to tell...etc. etc.

### Topic 7 (883 docs)

Top words: dark souls, dead cells, cells, souls, rip, skyrim, dark, elden ring, elden, west

- `1016950_4196.json review=12 sentence=sentence_7`: Без этого невозможно представить адекватную соревновательную часть и закрытые лиги.
- `1016950_4196.json review=12 sentence=sentence_16`: И это сбивает весь наработанный автоматизм, к тому же в ситуациях с "вилкой" решений, у тебя буквально теперь нет выбора.
- `1029780_17642.json review=6 sentence=sentence_15`: Пусть даже все это проходило бы одновременно, было бы уже хорошо т.к. нельзя поставить пол, пока крышу не снесли.

### Topic 8 (839 docs)

Top words: fps, settings, pc, performance, cpu, high, frame, stuttering, running, frames

- `1009290_6672.json review=3 sentence=sentence_122`: 越想越来气，我2060+I7，玩你这个游戏，先不说你画质，为什么要我在帧数上受那么大委屈？
- `1009290_6672.json review=3 sentence=sentence_123`: 重点是，你的画质远远不如3A大作啊，我这配置平时3A大作全开都没问题，你几年前的屎一样的画质，敢情把那些主流3A大作秒得渣都不剩？！
- `1009290_6672.json review=7 sentence=sentence_7`: 1. 프레임 저하 현상과 잦고 긴 로딩FHD에서 GTX 1080기준 모든 그래픽옵션을 High로 할 때 30프레임까지 떨어집니다.

### Topic 9 (765 docs)

Top words: weapons, weapon, guns, gun, ammo, use, shotgun, armas, different, waffen

- `1009290_6672.json review=5 sentence=sentence_13`: So you start off with your one-handed sword skills, but in total there are 9 different weapon types (according to Google- so like bows, warhammers, broadsword, etc) to play and for the most part they're fairly different from one another.
- `1012790_10355.json review=4 sentence=sentence_15`: i should mention that the way your character and his clumsy hands hold your weapon is rotated incorrectly across different headsets, if your on valve index or The Vive, or oculus or whatever the gun WILL NOT ALIGN with your controllers correctly.
- `1012790_10355.json review=4 sentence=sentence_19`: Guns, there are a lot of em, from pistols to shotguns to snipers to rifles, all with their own ammo type(accurately named mind you, so if yall Gun Nuts remember what caliber a M16A1 fires, good for you) and even more different TYPES of ammo, not only do we got Regular ♥♥♥♥ like FMJbut also AP, and Surplus, FMJ is fine, AP removes entities like Mr. Clean on Speed whilst cleaning your kitchen, but its about the same price as Speed which can make...

### Topic 10 (685 docs)

Top words: steam, ea, origin, launcher, launch, files, account, install, game steam, download

- `1009290_6672.json review=3 sentence=sentence_1`: 艾玛，今天呢，游戏发行不到三个月就开始打折咯~
- `1009290_6672.json review=7 sentence=sentence_9`: 그런데 싱글 게임이 주인 이런 게임에서 왜 굳이 EAC를 적용했는지는 의문입니다.
- `1009290_6672.json review=8 sentence=sentence_16`: インストール先にある「sao_al.exe」のプロパティを開き、「管理者として実行」にマーク。

### Topic 11 (654 docs)

Top words: sale, price, worth, buy, preo, wait, money, precio, cheaper, worth price

- `1012790_10355.json review=15 sentence=sentence_2`: the devs are offering this on the cheap.
- `1016950_4196.json review=3 sentence=sentence_54`: Normally I’d be alright with this but these prices are ridiculous.
- `1016950_4196.json review=4 sentence=sentence_7`: Oh, but you’re here to know if it is worth the money and time investment?

### Topic 12 (651 docs)

Top words: xcom, souls, rpg, soulslike, dark souls, dark, gears, like, games, roguelike

- `1009290_6672.json review=5 sentence=sentence_24`: Skyrim, when you think of it, also plays like a hack-and-slash - there is actually more strategy in SAO when it comes to fights than Skyrim.
- `1009290_6672.json review=7 sentence=sentence_30`: 근데 과연 이게 액션 어드벤쳐 RPG 일까요?
- `1009290_6672.json review=7 sentence=sentence_48`: 보통의 RPG 게임들이라면 무기/방어구 상점이 마주보거나 바로 옆에 있어야 한다는 기본 개념마저 버린 게임이니까요.

### Topic 13 (562 docs)

Top words: building, farming, buildings, build, farm, base, crops, houses, und, trees

- `1016800_13368.json review=4 sentence=sentence_14`: "Get party members to like you" and half Fallout 'Build a Comfortable, Functional Base".
- `1016800_13368.json review=9 sentence=sentence_15`: -> auch ein Basebuilding ist mit im Spiel integriert und gut umgesetzt - hier kann man zwar noch nicht soooo viel machen, aber die Idee sich seine eigenen Base zu bauen ist schon genial gemacht :)
- `1016800_13368.json review=10 sentence=sentence_22`: Nach dem Intro besitzt man eine Basis, die sich nach und nach ausbauen lässt.

### Topic 14 (522 docs)

Top words: dlc, dlcs, pack, edition, price, content, deluxe, packs, buy, base game

- `1009290_6672.json review=6 sentence=sentence_1`: 9月中旬，500块买了豪华版，按捺住怒火玩到现在，达成了全成就，可以来写篇评测了。
- `1016950_4196.json review=5 sentence=sentence_19`: I'm still mad about paying full price for BB3 and having teams parted out to me as paid DLC.
- `1016950_4196.json review=15 sentence=sentence_8`: Et comme un fan naïf j'ai acheté la version "brutal" en me disant que je pourrais peut être avoir plus de races, de DLC, de Star Players ou de personnalisation ...

### Topic 15 (500 docs)

Top words: map, maps, mapa, mapas, die, markers, und, world, way, open

- `1012790_10355.json review=3 sentence=sentence_18`: As a student in game design myself, I can attest that is NO small task to do, throwing out months of work building, testing, and refining an ENTIRE WORLD MAP simply to make it better.
- `1012790_10355.json review=3 sentence=sentence_20`: Now the world map is split into sections, all of which have multiple entry points (already an improvement), interconnect to each other, and some of which connect directly to your base.
- `1012790_10355.json review=3 sentence=sentence_23`: And even more than that, maps can change according to you mission to add even more, different varieties to you travels.

### Topic 16 (491 docs)

Top words: ror, goh, ps2, heat, dirt, nier automata, automata, melty, ryu, nier

- `1016950_4196.json review=12 sentence=sentence_45`: Я свою копию получил за победу в командном чемпионате мира, поэтому я с этой подводной лодки деться никуда не могу, но вам я крайне не рекомендую СЕЙЧАС брать игру.
- `1029780_17642.json review=6 sentence=sentence_2`: Были моменты, которые в игре лишние (имхо конечно) и которых не хватило.
- `1030210_11120.json review=4 sentence=sentence_11`: Разработчики каким-то абсолютно невероятным образом к х*ям ломают то, что работало раньше и начинают медленно чинить то, что уже было чинено и (казалось бы!) "работает - не трогай", но нет.

### Topic 17 (481 docs)

Top words: combat, fun, fighting, battle, fights, fight, mechanics, really, game combat, battles

- `1012790_10355.json review=13 sentence=sentence_23`: I'm too young for the Vietnam War, but this SOUNDS like how I imagine it must have FELT.
- `1016800_13368.json review=7 sentence=sentence_48`: 敵兵のアルゴリズムも複雑ではないので、後半になるとコソコソするのが面倒になってしまう。
- `1029690_19693.json review=7 sentence=sentence_7`: Authentic.-I NEVER engage in combat and play every mission undetected.

### Topic 18 (478 docs)

Top words: graphics, art, animations, models, visuals, grafik, style, beautiful, grficos, pixel

- `1009290_6672.json review=7 sentence=sentence_5`: 애니메이션 캐릭터의 매력에 빠지게 되면움직이지도 못하는 10만원 넘는 피규어까지 구입하는데,하물며 자신이 좋아하는 캐릭터를 직접 움직이고 그 스토리를 즐기는 건 엄청난 매력적인 부분이죠.
- `1009290_6672.json review=8 sentence=sentence_20`: 描画方式はどちらでもよいが「フルスクリーン」の方が微妙に軽い。
- `1012790_10355.json review=13 sentence=sentence_8`: The visuals of the zone, rather than making my mind float in a happy narcotic bliss, made my brain want to smash itself with a brick.

### Topic 19 (468 docs)

Top words: price, sale, worth, game, recommend, preo, jogo, buy, game worth, price tag

- `1012790_10355.json review=11 sentence=sentence_13`: BEFORE YOU BUY THIS GAME -- research it.
- `1016800_13368.json review=3 sentence=sentence_15`: Either way, this is a great game on its own, especially for the price.
- `1016800_13368.json review=6 sentence=sentence_26`: I was idly curious about this game initially, and bought in as part of the effort to raise money for ukraine, but MAN i was not expecting this level of quality.

### Topic 20 (367 docs)

Top words: dlc, jrpg, ost, rpg, katana, fortune, deaths, bgm, ai, ex

- `1009290_6672.json review=7 sentence=sentence_3`: 혹시 본인이 소아온 애니메이션을 재밌게 봐서 구입을 원하신다면, 일반적인 JRPG 장르의 게임이라 생각하고 구입하시는 거라면,실망하실 각오 미리 단단히 하시길 바랍니다.
- `1009290_6672.json review=7 sentence=sentence_4`: 분명 이 게임은 소아온 원작 노벨/애니팬들에겐 괜찮은 작품입니다.
- `1009290_6672.json review=7 sentence=sentence_25`: 비주얼 노벨에 RPG 시스템만 살짝 얹어놓은듯한 장르챕터 2까지 스킵 한번 안하고 의뢰/유물 퀘스트 적당히 하면서 느낀거지만, 이 게임은 JRPG를 좋아해서 구입하려 하는 유저들이라면 절대적으로 피해야 할 게임입니다.

### Topic 21 (355 docs)

Top words: story, plot, stories, good story, interesting, written, good, predictable, ending, storyline

- `1016800_13368.json review=3 sentence=sentence_27`: (With a Fallout-esque tapestry ending and everything)
- `1016800_13368.json review=7 sentence=sentence_57`: 個々の要素はコンパクトだがストーリーを軸に手堅くまとまっており、完成度が高い。
- `1016800_13368.json review=9 sentence=sentence_11`: Die Story von Igor und Tatyana wird spannen umgesetzt und fordert einen stehts weiter zu machen um ihr wieder näher zu kommen->

### Topic 22 (352 docs)

Top words: review, reviews, negative, write, comments, positive, read, helpful, change review, im

- `1016800_13368.json review=3 sentence=sentence_3`: (Which might be the reason for the negative reviews.
- `1016950_4196.json review=7 sentence=sentence_21`: They need to earn our trust back we MUST wait for reviews.
- `1016950_4196.json review=10 sentence=sentence_1`: At this point a review from me is pointless.

### Topic 23 (332 docs)

Top words: enemies, enemy, attacks, dodge, attack, units, hit, damage, gegner, opponent

- `1009290_6672.json review=5 sentence=sentence_31`: You get to block incoming attacks, buff allies, etc.
- `1012790_10355.json review=3 sentence=sentence_14`: You almost fight exclusively like 2 enemies ever, which is a shame cause there's a whole lot of them to use and fighting against some of them just becomes a hassle in environments that make fighting either way too close or way too far (staying vague so I don't spoil enemy discovery cause that's a lot of the fun).
- `1012790_10355.json review=6 sentence=sentence_35`: Knowing that all the enemies I've killed will come back when The Tide is over.

### Topic 24 (328 docs)

Top words: devs, developers, updates, patch, dev, update, entwickler, feedback, fix, developer

- `1012790_10355.json review=3 sentence=sentence_15`: And so what did the devs do in response to these significant, fundamental design problems.
- `1016800_13368.json review=6 sentence=sentence_29`: I really hope the Dev team and studio arent impacted by world events and can continue to make new material!
- `1016800_13368.json review=9 sentence=sentence_13`: man hat echte Fotos von den Entwicklern, die in Tschernobyl gemacht wurden ;)

### Topic 25 (310 docs)

Top words: boss, bosses, fight, boss fights, fights, final, patterns, beat, fun, attack

- `1036890_4908.json review=11 sentence=sentence_13`: Actually has some of the better bosses in the series, but there's only 2 this time.
- `1056640_23530.json review=5 sentence=sentence_13`: Then when emergency quests happen and multiparty bosses appear, those are really fun too!
- `1056640_23530.json review=5 sentence=sentence_18`: A problem can be that bosses aren't all that fun to fight with too many players, since they flinch until dead.

### Topic 26 (307 docs)

Top words: una, minutos, la, para, potato, las, hasta, luego, el, en

- `1041720_6700.json review=14 sentence=sentence_50`: Maybe heat up a can of refried beans or something".
- `1057090_26450.json review=12 sentence=sentence_11`: You can run it on a microwave✅ Average✅ High end
- `1063660_11863.json review=14 sentence=sentence_6`: Potato☐ Mínimo☑ Decente☑ Rápido☐

### Topic 27 (304 docs)

Top words: controller, mouse, controls, keyboard, control, controllers, xbox, teclado, keys, support

- `1009290_6672.json review=8 sentence=sentence_38`: Steam側でのコントローラー設定を強制ONにしないと、Steam側でのコントローラーサポート機能が動作しない。
- `1029690_19693.json review=9 sentence=sentence_8`: Oyunu 1.5 saat kadar kontrolcüyle oynadım fakat kontrolcü entegrasyonu hiç iyi değil ayrıca hem silah hem itemleri aynı silah çarkına koymakda pek akıllıca değil.-
- `1029780_17642.json review=9 sentence=sentence_21`: ich platze vor Genervt-sein, keine Blueprintfunktion, Tastenbelegung ist rudimetär)- Hitzewellen, Kältewellen, Thors Donner etc sind NICHT anwählbar.

### Topic 28 (301 docs)

Top words: coop, multiplayer, play, online, friends, mode, players, works, experience, single player

- `1049410_23830.json review=12 sentence=sentence_24`: Мультиплеер я не пробовал, но говорят это штука веселая.
- `1062520_16450.json review=4 sentence=sentence_3`: .....however me and my gf were a little disappointed that the multiplayer doesnt give us the full coop feeling yet - you can visit each others islands but cant progress in the town quest together.
- `1062520_16450.json review=4 sentence=sentence_10`: and then we start playin.i know that the developer is looking into multiplayer progression, so no need for a refund, no need for a rant.

### Topic 29 (297 docs)

Top words: early access, early, access, game early, release, beta, released, game, content, title

- `1012790_10355.json review=8 sentence=sentence_6`: 楽しめたならbeta branchから1.0もプレイして欲しいです。
- `1016800_13368.json review=9 sentence=sentence_4`: Bei Chernobylite kann man sehr gut sehen, wie EarlyAcces gut und informativ funktionieren kann - es gibt wöchentliche Updates der Entwickler und auch auf dem Youtube-Channel kann man den Fortschritt sehr gut mit verfolgen.
- `1016950_4196.json review=9 sentence=sentence_5`: At release it was in early, early beta phase at best (and that's a very generous description).
