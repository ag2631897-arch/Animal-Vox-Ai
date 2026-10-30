"""
AnimalVox AI — Bioacoustic Natural Language Generator (SRS v1.1)
================================================================
Translates detected animal vocalizations and ethological biomarkers into authentic,
first-person, scientifically grounded human language.

Supports all 12 ethological behaviors across all 6 species groups (Dogs, Cats,
Birds, Dolphins, Whales, Frogs), modulated by:
- Intensity & Urgency (Low, Medium, High, Critical)
- Emotional state from Module 03 (Loneliness, Contentment, Stress, Fear, etc.)
- Sequence escalation context
"""

from typing import Dict, Optional


ETHOLOGY_TRANSLATION_MAP = {
    "dogs": {
        "feeding_call": {
            "name": "Food Solicitation Bark / Whine",
            "high": "I smell food and I'm starving! Please give me some right now! Look at me, I've been such a good dog!",
            "medium": "Hey, are you eating something good? You know I'd love a bite if you're willing to share!",
            "low": "Mmm, smells nice over there. Any chance there's a little treat heading my way?",
            "context": "Canine solicitation vocalization expressing food anticipation and appetite. High repetition barks indicate eager begging (Bradshaw 2011; Yin 2002; Feddersen-Petersen 2000)."
        },
        "alarm_ground": {
            "name": "Ground Intruder Alert Bark",
            "high": "Stay away from here. I'm on guard and warning you not to come closer.",
            "medium": "Something is too close to our space. I'm staying alert.",
            "low": "I'm letting you know that something nearby has caught my attention.",
            "context": "Territorial alert bark alerting human pack to nearby approach of an unfamiliar individual or animal (Feddersen-Petersen 2000; Yin 2002)."
        },
        "alarm_aerial": {
            "name": "Aerial Vigilance Bark",
            "high": "LOOK UP! Strange shadow or big bird right overhead! Watch out above us!",
            "medium": "Something big is flying over our yard. Keeping my eyes locked on it.",
            "low": "Just tracking something high up in the sky.",
            "context": "Vigilance bark triggered by rapid movement or large silhouettes overhead (Bradshaw 2011)."
        },
        "territorial_warning": {
            "name": "Territorial Boundary Bark",
            "high": "This is my territory. Stay back and do not come any closer.",
            "medium": "You are too close to my space. Please back away.",
            "low": "This area is mine, so please give me some room.",
            "context": "Assertive territorial defense vocalization signaling property ownership and pack defense (Bradshaw 2011)."
        },
        "mating_call": {
            "name": "Broadcast Mating Howl",
            "high": "I'm searching for a companion! Can any dogs in the neighborhood hear me calling?!",
            "medium": "Calling out across the distance to see if any other friendly dogs are nearby.",
            "low": "Just singing out a gentle tune across the evening breeze.",
            "context": "Long-distance reproductive broadcast howl intended to announce reproductive readiness across territory (Harrington & Asa 2003)."
        },
        "contact_call": {
            "name": "Contact Whine / Separation Call",
            "high": "WHERE DID EVERYONE GO?! I'm all alone! Please come back to me right now, I need you!",
            "medium": "Hello? Anyone home? I'm looking for you, where did you go?",
            "low": "Just checking in to make sure you're still nearby in the house.",
            "context": "Affiliative contact vocalization expressing attachment and desire for physical proximity to caregiver (Overall 2013)."
        },
        "distress_call": {
            "name": "Distress Whimper / Pain Yelp",
            "high": "PLEASE HELP ME! I'm in pain or terrified! Something is really wrong, don't leave me!",
            "medium": "I'm really not feeling okay about this situation. Can you please stay close to comfort me?",
            "low": "I'm feeling a little anxious and uneasy right now. A gentle hug would really help.",
            "context": "Distress vocalization indicating physical pain, severe fear, or acute separation distress (Bekoff 2001; Overall 2013)."
        },
        "play_vocalization": {
            "name": "Play Solicitation Bow-Bark",
            "high": "PLAY! PLAY! PLAY! I can't CONTAIN myself! Throw the ball right now! Chase me! DO SOMETHING!",
            "medium": "Hey, wanna play? I'm in the mood for some fun! Grab a toy and let's go!",
            "low": "I could go for a little game if you're up for it. I'm feeling playful.",
            "context": "Play solicitation call characterized by uneven pitch contours, play bows, and bouncy rhythm (Bekoff 2001)."
        },
        "submission_signal": {
            "name": "Appeasement Submissive Whimper",
            "high": "Please don't be mad! I surrender completely! I'm no threat to you at all!",
            "medium": "Okay, okay — you're the boss. I'll back down. No trouble from me at all.",
            "low": "Hey, just being friendly. No competition here, we're all good.",
            "context": "Deference and appeasement vocalization intended to disarm social tension and avoid conflict (Yin 2002)."
        },
        "aggression_warning": {
            "name": "Low Threat Growl",
            "high": "Back away now. I need space and I may defend myself if you keep approaching.",
            "medium": "I am uncomfortable with you this close. Please give me space.",
            "low": "I need a little more space right now.",
            "context": "Direct threat display warning of defensive aggression if social distance is breached (Bradshaw 2011; Feddersen-Petersen 2000)."
        },
        "social_bonding": {
            "name": "Affectionate Pack Grunt / Soft Howl",
            "high": "I love you so much! You are my absolute favorite human in the whole world!",
            "medium": "I'm so glad we're together right now. You make me feel completely safe and happy.",
            "low": "Just relaxing here next to you. Life is so good with you around.",
            "context": "Social bonding and pack affiliation vocalization reinforcing attachment bonds (Bradshaw 2011)."
        },
        "navigation_echolocation": {
            "name": "Exploratory Scent Chuff",
            "high": "I've picked up a hot scent trail! Following it with full concentration!",
            "medium": "Sniffing out something really interesting here on the ground. Checking it out!",
            "low": "Just exploring the yard and checking out recent scent marks.",
            "context": "Olfactory scanning and exploratory vocal burst during trail investigation (Bradshaw 2011)."
        }
    },
    "cats": {
        "feeding_call": {
            "name": "Solicitation Meow (Feeding)",
            "high": "MY BOWL IS EMPTY! Can you not see I am literally withering away?! FEED ME RIGHT NOW!",
            "medium": "Excuse me, human. It is past my mealtime. Please fill the bowl without delay.",
            "low": "A little treat would be acceptable right about now if you have one.",
            "context": "High-frequency solicitation meow evolved specifically to prompt human caregiving and food provisioning (Nicastro 2004; McComb et al. 2009)."
        },
        "distress_call": {
            "name": "Distress Caterwaul",
            "high": "HELP! I'm trapped, hurt, or panicking! Come rescue me immediately!",
            "medium": "I am deeply uncomfortable and anxious. Please come check on me.",
            "low": "A little unsettled. Please stay nearby where I can see you.",
            "context": "Acoustic signature of feline acute distress and pain (Turner & Bateson 2014)."
        },
        "social_bonding": {
            "name": "Contentment Purr",
            "high": "Purrrrr... this is absolute perfection. Don't move a single muscle, just keep petting me.",
            "medium": "I feel safe and content curling up next to you right now.",
            "low": "A gentle purr to let you know we're on good terms.",
            "context": "Low-frequency rhythmic purring (25-50 Hz) signaling deep contentment and human-cat bonding (Turner & Bateson 2014)."
        },
        "aggression_warning": {
            "name": "Defensive Hiss & Spat",
            "high": "HISSSS! STAY AWAY FROM ME! My claws are out and I WILL strike with full force!",
            "medium": "I'm hissing and growling. One more inch and you get a claw.",
            "low": "Back off, my tail is twitching and my patience is thin.",
            "context": "Defensive threat display signalling imminent scratch or bite if cornered (Crowell-Davis 2004)."
        },
        "play_vocalization": {
            "name": "Playful Chirrup / Trill",
            "high": "ZOOMIES ENGAGED! Look at me attack this imaginary ghost! Catch me if you can!",
            "medium": "Wiggle that string! Let me pounce on it right now!",
            "low": "I might swat at that toy if you dangle it just right.",
            "context": "Playful predatory practice chirrup and burst vocalization (Bradshaw 2013)."
        },
        "alarm_ground": {
            "name": "Startle Warning Growl",
            "high": "DANGER! Something strange and dangerous is on the floor! Watch out down there!",
            "medium": "Unfamiliar creature or object detected. Observing cautiously from a safe height.",
            "low": "Keeping a watchful eye on that strange movement down there.",
            "context": "Vigilance and alarm reaction to ground-level anomalies (Turner & Bateson 2014)."
        },
        "alarm_aerial": {
            "name": "Predator Alert Chitter",
            "high": "SHADOW ABOVE! Duck and take cover immediately!",
            "medium": "A large bird just flew over the window. Staying crouched low.",
            "low": "Tracking a flying creature overhead.",
            "context": "Visual predator alert and chattering response to overhead movement (Nicastro 2004)."
        },
        "territorial_warning": {
            "name": "Territorial Yowl",
            "high": "GET OUT OF MY TERRITORY! This window and room belong exclusively to ME!",
            "medium": "You are way too close to my domain. Turn around and walk away.",
            "low": "Consider this boundary established. Don't test me.",
            "context": "Territorial boundary announcement to deter neighboring felines (Bradshaw 2013)."
        },
        "mating_call": {
            "name": "Mating Yowl",
            "high": "HEAR ME, potential mates! I am magnificent and available! Who dares approach?!",
            "medium": "Chattering and calling... searching for a worthy companion.",
            "low": "Broadcasting my presence into the night breeze.",
            "context": "Audible courtship broadcast common during estrus or breeding arousal (Turner & Bateson 2014)."
        },
        "contact_call": {
            "name": "Locational Contact Trill",
            "high": "MRRROOWW! Where did you go?! You are supposed to be in my sight at all times!",
            "medium": "Hello? Human? Which room are you in? Come show yourself.",
            "low": "Just chirping to make sure you haven't abandoned me.",
            "context": "Affiliative trill/meow used to track human companion location (Nicastro 2004)."
        },
        "submission_signal": {
            "name": "Submissive Throat Click",
            "high": "I'm rolling on my back — don't hurt me! Peace! I yield completely!",
            "medium": "I'm lowering my ears. Let's not escalate this.",
            "low": "Just keeping low and deferring to you.",
            "context": "Feline appeasement posture and soft throat vocalization (Bradshaw 2013)."
        },
        "navigation_echolocation": {
            "name": "Hunting Whisker Chatter",
            "high": "Creeping silently... stalking every slight vibration in the floorboards.",
            "medium": "Patrolling the perimeter and investigating new scents.",
            "low": "Just casually inspecting my kingdom.",
            "context": "Excited chattering and jaw-chattering during hunting focus (Turner & Bateson 2014)."
        }
    },
    "birds": {
        "alarm_aerial": {
            "name": "Aerial Predator Alarm (Seet Call)",
            "high": "HAWK! HAWK OVERHEAD! Freeze or dive into the thick bushes right now!",
            "medium": "Large raptor gliding above us. Keep quiet and stay under cover.",
            "low": "High flyer overhead. Staying vigilant in the foliage.",
            "context": "Acoustic 'seet' alarm call triggering instant freeze response across mixed-species flocks (Morse 1980; Marler 1955)."
        },
        "alarm_ground": {
            "name": "Ground Predator Mobbing Call",
            "high": "CAT ON THE GROUND! Mobbing call! Everybody join in and drive it away!",
            "medium": "Predator spotted in the grass below. Calling to warn the flock.",
            "low": "Suspicious movement near the trunk. Keeping high in the canopy.",
            "context": "Harsh, broad-band mobbing call designed to recruit flock members to harass terrestrial predators (Marler 1955)."
        },
        "feeding_call": {
            "name": "Foraging Attraction Call",
            "high": "FOOD FOUND! Ripe berries and seeds everywhere! Come eat with the flock!",
            "medium": "Good foraging spot discovered here. Enough food for everyone.",
            "low": "Found a little snack in the branches. Feeding peacefully.",
            "context": "Food-associated call signaling foraging patch quality to conspecifics (Catchpole & Slater 2008)."
        },
        "social_bonding": {
            "name": "Pair-Bond Duet / Preening Call",
            "high": "Singing together in perfect harmony! Our pair bond is unbreakable!",
            "medium": "Gentle contact chirp while we preen each other in the shade.",
            "low": "Soft affiliative note. All is well in our territory.",
            "context": "Duetting and affiliative vocalizations reinforcing pair bonds (Catchpole & Slater 2008)."
        },
        "mating_call": {
            "name": "Dawn Chorus Advertisement Song",
            "high": "Listen to my brilliant song! I hold the best territory and the strongest vigor in the canopy!",
            "medium": "Singing from my favorite branch to attract a partner.",
            "low": "Soft territorial melody at sunset.",
            "context": "Complex breeding song demonstrating male fitness and territory quality (Catchpole & Slater 2008)."
        },
        "contact_call": {
            "name": "Flock Flight Call",
            "high": "Keep together! Don't lose the flock! We are turning north!",
            "medium": "Chirp-chirp! Checking positions across the trees.",
            "low": "Quiet contact peep. I am right on your wing.",
            "context": "Short flight calls maintaining flock cohesion in transit (Catchpole & Slater 2008)."
        },
        "distress_call": {
            "name": "Screech / Distress Yelp",
            "high": "CAUGHT! HELP! Struggling against a predator! Alarm!",
            "medium": "Injured or trapped! Calling out in distress!",
            "low": "Frightened and disoriented. Calling for assistance.",
            "context": "Loud, broadband distress scream aimed at startling predators into releasing prey (Marler 1955)."
        },
        "territorial_warning": {
            "name": "Territorial Counter-Singing",
            "high": "THIS IS MY TREE! Cross that branch and I will dive-bomb you!",
            "medium": "Answering your song across the hedge: keep to your side.",
            "low": "Singing to confirm my boundary line.",
            "context": "Vocal territory defense and countersinging between rival neighbors (Catchpole & Slater 2008)."
        },
        "play_vocalization": {
            "name": "Aerobatic Joy Call",
            "high": "Rolling through the updraft! Diving and swooping just for the thrill of flight!",
            "medium": "Chasing leaves and darting through the twigs in play.",
            "low": "Lighthearted chirping while fluttering between perches.",
            "context": "Vocalizations associated with non-foraging aerobatic flight and juvenile social play."
        },
        "submission_signal": {
            "name": "Perch Appeasement Flutter",
            "high": "I yield the perch! Lowering crest and fluttering wings!",
            "medium": "Backing down from the feeder. You eat first.",
            "low": "Deferring to dominant flock mate.",
            "context": "Submissive display and posture to reduce aggression at feeding sites."
        },
        "aggression_warning": {
            "name": "Bill-Snap / Hiss Threat",
            "high": "SNAP! BACK OFF! Get away from my nest or face my beak!",
            "medium": "Hissing warning: you are uncomfortably close to my eggs.",
            "low": "Threat chatter to warn away an intruder.",
            "context": "Direct nest-defense aggressive display with bill-snapping (Marler 1955)."
        },
        "navigation_echolocation": {
            "name": "Echolocation Click / Spatial Chirp",
            "high": "Rapid clicks piercing the dark cave! Mapping the stone walls!",
            "medium": "Acoustic spatial ping checking the canopy density ahead.",
            "low": "Quiet spatial orientation note.",
            "context": "Auditory spatial mapping utilized by swiftlets and oilbirds in low light (Brinklov 2013)."
        }
    },
    "dolphins": {
        "contact_call": {
            "name": "Signature Whistle",
            "high": "It's me! I'm over here by the reef! Pod, where are you?! Reconnect with me!",
            "medium": "Signature whistle broadcasting my identity. Checking in with the pod.",
            "low": "Gentle identity whistle. Swimming nearby.",
            "context": "Individual signature whistle functioning as an acoustic name for pod coordination (Janik & Sayigh 2013)."
        },
        "navigation_echolocation": {
            "name": "High-Frequency Echolocation Click Train",
            "high": "Target locked! Rapid click train imaging fish school 20 meters ahead!",
            "medium": "Scanning the seabed contours and depth with steady click pulses.",
            "low": "Ambient spatial navigation clicks.",
            "context": "Directional ultrasonic click trains for biosonar spatial mapping and prey acquisition (Au 1993)."
        },
        "feeding_call": {
            "name": "Coordinated Fish-Herding Whistle",
            "high": "Bait ball surrounded! Tighten the circle, then take turns striking!",
            "medium": "Fish school concentrated. Feeding together with the pod.",
            "low": "Foraging calls as we browse the shallows.",
            "context": "Acoustic communication coordinating cooperative pack-hunting and prey containment (Herzing 2011)."
        },
        "social_bonding": {
            "name": "Affiliative Synchronous Whistle",
            "high": "Swimming flank-to-flank in total harmony! Deep pod bond!",
            "medium": "Gentle pectoral fin contact and shared tonal whistling.",
            "low": "Relaxed synchrony. Contentment in the pod.",
            "context": "Low-amplitude affiliative calls associated with physical contact and alliance formation (Herzing 2011)."
        },
        "play_vocalization": {
            "name": "Bow-Riding Kinetic Burst",
            "high": "SURFING THE SWELL! Leaping high out of the water! Pure kinetic joy!",
            "medium": "Chasing each other through the waves in playful tag.",
            "low": "Bouncy whistles while playing with sea grass.",
            "context": "Playful vocalizations accompanying non-foraging acrobatic behaviors and object play."
        },
        "alarm_ground": {
            "name": "Apex Predator Alarm (Shark/Orca Alert)",
            "high": "SHARK DETECTED! Form defensive perimeter around calves immediately!",
            "medium": "Large predatory shape spotted in deep water. High alert.",
            "low": "Caution note: keep calves close.",
            "context": "Rapid burst pulse alarm triggering defensive nursery group formation (Janik 2013)."
        },
        "alarm_aerial": {
            "name": "Surface Anomaly Alert",
            "high": "Boat hull approaching at high speed! Dive deep right now!",
            "medium": "Vessel sound overhead. Moving away from propeller zone.",
            "low": "Monitoring surface turbulence.",
            "context": "Avoidance whistles and rapid dives in response to motorized craft."
        },
        "distress_call": {
            "name": "Entanglement / Separation Distress Whistle",
            "high": "ENTANGLED OR INJURED! I cannot reach the surface! Help me!",
            "medium": "Separated from mother in murky water. High distress calls.",
            "low": "Fatigued and calling for support.",
            "context": "Urgent distress whistles prompting cooperative rescue behavior by pod mates (Connor 2000)."
        },
        "territorial_warning": {
            "name": "Jaw-Clap Threat",
            "high": "JAW CLAP! Back away from this feeding zone or we will ram you!",
            "medium": "Burst-pulse buzzing warning an unfamiliar alliance to maintain distance.",
            "low": "Territorial presence reminder.",
            "context": "Percussive jaw-claps and burst pulses signalling aggressive intent (Herzing 2011)."
        },
        "aggression_warning": {
            "name": "Burst-Pulse Threat Buzz",
            "high": "LOUD BURST-PULSE! Back off immediately before physical confrontation!",
            "medium": "Aggressive posture and buzzing sound. Give way.",
            "low": "Mild irritation buzz.",
            "context": "Acoustic agonistic signaling during dominance disputes (Herzing 2011)."
        },
        "mating_call": {
            "name": "Courtship Serenade / Pop Train",
            "high": "Synchronized courtship dance! Swimming together in tight spirals!",
            "medium": "Rhythmic pop trains produced by male alliance courting a female.",
            "low": "Courting whistles in the lagoon.",
            "context": "Rhythmic multi-male alliance courtship vocalizations (Connor et al. 2000)."
        },
        "submission_signal": {
            "name": "Appeasement Chirp",
            "high": "Yielding position! No conflict intended, rolling sideways.",
            "medium": "Deferring to dominant alliance. Moving aside.",
            "low": "Passive compliance chirp.",
            "context": "Submissive avoidance whistles during inter-alliance encounters."
        }
    },
    "whales": {
        "mating_call": {
            "name": "Breeding Song Theme",
            "high": "I am here, singing the great song across the entire ocean! Hear my strength!",
            "medium": "Melodic phrases repeating through the deep water, carrying for miles.",
            "low": "A slow, deep resonance resonating through the abyss.",
            "context": "Humpback male breeding song composed of rhyming themes that evolve each season (Payne & McVay 1971)."
        },
        "contact_call": {
            "name": "Low-Frequency Infrasonic Moan",
            "high": "Calling across the ocean basin! Pod members 50 miles away, can you hear me?!",
            "medium": "Deep 20 Hz pulse traveling hundreds of miles through the SOFAR channel.",
            "low": "Quiet infrasound rumble. Maintaining contact across the migration route.",
            "context": "Blue / Fin whale 20 Hz pulses engineered for trans-oceanic acoustic propagation (Payne & Webb 1971)."
        },
        "social_bonding": {
            "name": "Mother-Calf Acoustic Whisper",
            "high": "I am right beneath you, little one. I will lift you to the surface to breathe.",
            "medium": "Soft, low-amplitude grunts ensuring mother and calf stay touching in the surf.",
            "low": "Gentle rumble of comfort during nursing.",
            "context": "Cryptic low-amplitude 'whispers' between mother and calf to avoid attracting killer whales (Videsen et al. 2017)."
        },
        "feeding_call": {
            "name": "Bubble-Net Feeding Cry",
            "high": "BUBBLE NET RISING! All whales surge upwards together into the krill ball NOW!",
            "medium": "Ascending trumpet call coordinating the bubble net spiral.",
            "low": "Feeding groan as we dive beneath the herring swarm.",
            "context": "Cooperative bubble-net feeding trumpet blast freezing fish inside a ring of bubbles (Jurasz & Jurasz 1979)."
        },
        "distress_call": {
            "name": "Stranding / Entanglement Pulse",
            "high": "SHALLOW WATER TRAP! I cannot navigate these sandbars! High distress!",
            "medium": "Heavy labored breathing and acoustic clicks in shallow surf.",
            "low": "Tired, disoriented pulse.",
            "context": "Acoustic panic signals emitted during disorientation or acoustic sonar interference."
        },
        "alarm_ground": {
            "name": "Orca Predator Warning",
            "high": "KILLER WHALES APPROACHING! Form protective circle around all calves!",
            "medium": "Stealth mode: cease singing immediately. Predators in the acoustic range.",
            "low": "Silent gliding. Danger nearby.",
            "context": "Anti-predator silence and protective grouping triggered by transient orca vocalizations (Ford 1989)."
        },
        "alarm_aerial": {
            "name": "Surface Obstacle Warning",
            "high": "Cargo vessel directly on our path! Sound the breach alarm!",
            "medium": "Loud engine noise ahead. Diving deep to avoid vessel strike.",
            "low": "Monitoring ship noise on the surface.",
            "context": "Acoustic avoidance of commercial shipping lanes."
        },
        "play_vocalization": {
            "name": "Breaching Slap & Trumpet",
            "high": "BREACHING! 40 tons leaping into the sky! Thunderous water slap!",
            "medium": "Lobtailing and flipper slapping the surface in rhythmic play.",
            "low": "Playful rolling in the kelp forest.",
            "context": "Percussive surface breaches used for both acoustic communication and energetic play."
        },
        "territorial_warning": {
            "name": "Competitive Display Blast",
            "high": "This migration lane is mine! Back off from this escort position!",
            "medium": "Loud underwater bellow warning away competing male escorts.",
            "low": "Deep rumble establishing space.",
            "context": "Acoustic competition during competitive breeding pods (Tyack 1981)."
        },
        "aggression_warning": {
            "name": "Bubble Trail & Jaw Strike",
            "high": "Forceful jaw strike! Keep away from this female or prepare to clash!",
            "medium": "Blowing a wall of bubbles to establish dominant escort position.",
            "low": "Sub-surface thud asserting dominance.",
            "context": "Agonistic display between rival male humpbacks during escort competition."
        },
        "submission_signal": {
            "name": "Yielding Moan",
            "high": "Surrendering the escort lead. Falling back to trailing position.",
            "medium": "Quiet descent, yielding the territory.",
            "low": "Submissive course change.",
            "context": "De-escalation by subordinate males in competitive pods."
        },
        "navigation_echolocation": {
            "name": "Bathymetric Biosonar Ping",
            "high": "Seamount mapped! Echo returning from the underwater ridge 10 miles away!",
            "medium": "Bouncing sound off the continental shelf to guide our migration path.",
            "low": "Ambient bathymetric orientation pulse.",
            "context": "Low-frequency topographic navigation using ocean floor reverberation."
        }
    },
    "frogs": {
        "mating_call": {
            "name": "Advertisement Chorus",
            "high": "JUG-O-RUM! I am the loudest, strongest male in this pond tonight! Ladies, come to my station!",
            "medium": "Singing steadily from my lily pad. Ready for a partner to join me.",
            "low": "A gentle croak into the humid night.",
            "context": "Advertisement call demonstrating male body size and acoustic endurance (Duellman & Trueb 1994)."
        },
        "territorial_warning": {
            "name": "Encounter Croak",
            "high": "THIS LILY PAD IS MINE! Leap away before I push you into the water!",
            "medium": "You are crowding my call site. Find your own territory.",
            "low": "Short territorial pip establishing my perch.",
            "context": "Aggressive encounter call warning rival males away from calling perches (Wells 2007)."
        },
        "distress_call": {
            "name": "Release Scream",
            "high": "EEEEEK! LET ME GO! I am toxic and you cannot swallow me! Release me!",
            "medium": "Distress squawk as predator grabs leg! Struggling free!",
            "low": "Panicked release chirp.",
            "context": "Loud open-mouth distress scream intended to startle predators or attract a larger predator to interfere (Wells 2007)."
        },
        "alarm_ground": {
            "name": "Predator Splash Alarm",
            "high": "HERON ON THE BANK! PLOP! Dive into the deep mud and stay silent!",
            "medium": "Snake detected near the reeds. Cease chorusing and submerge.",
            "low": "Suspicious ripple. Sitting frozen at the water's edge.",
            "context": "Anti-predator silence waves propagating through the chorusing wetland (Wells 2007)."
        },
        "feeding_call": {
            "name": "Foraging Snap & Pip",
            "high": "Moth swarm right above the water! Snap! Catching dinner!",
            "medium": "Insects buzzing nearby. Tongue primed for the strike.",
            "low": "Watching beetles along the moss.",
            "context": "Feeding-associated clicks and swallowing sounds during insect emergence."
        },
        "social_bonding": {
            "name": "Synchronized Chorus Pulse",
            "high": "The entire wetland is singing as ONE! Thousands of voices synchronized!",
            "medium": "Joining the rhythmic chorus pulses across the marsh.",
            "low": "Humming along with neighboring frogs.",
            "context": "Lek-based synchronized chorus behavior confusing predator localization (Wells 2007)."
        },
        "play_vocalization": {
            "name": "Juvenile Splash Chirp",
            "high": "Leaping and tumbling across the duckweed! Summer rain celebration!",
            "medium": "Playful hop onto a floating log.",
            "low": "Gentle rain croak.",
            "context": "Vocalizations associated with post-metamorphic emergence and rain joy."
        },
        "submission_signal": {
            "name": "Release Squeak",
            "high": "I'm a MALE, not a female! Get off me! Wrong frog!",
            "medium": "Vibrating sides and release chirp to stop mistaken amplexus.",
            "low": "Quick release vibration.",
            "context": "Male-male release call preventing wasted reproductive amplexus (Wells 2007)."
        },
        "aggression_warning": {
            "name": "Wrestling Growl",
            "high": "Wrestling for the bank! Shoving match underway!",
            "medium": "Low guttural growl before a physical wrestling bout.",
            "low": "Aggressive posture chirp.",
            "context": "Physical territorial combat vocalizations between bullfrogs."
        },
        "alarm_aerial": {
            "name": "Owl / Bat Alert Silence",
            "high": "WING FLAP OVERHEAD! Plunge beneath the water lilies instantly!",
            "medium": "Shadow passed over the pond. Submerging until safe.",
            "low": "Holding breath under water.",
            "context": "Instantaneous cessation of chorusing when nocturnal aerial predators approach."
        },
        "contact_call": {
            "name": "Wetland Locator Peep",
            "high": "Peep-peep! Which way is the breeding pool? Following the moisture!",
            "medium": "Calling during migration to vernal pools.",
            "low": "Quiet leaf-litter peep.",
            "context": "Migratory locator calls directing amphibians toward ephemeral breeding ponds."
        },
        "navigation_echolocation": {
            "name": "Moisture & Bank Orientation Pulse",
            "high": "Following the acoustic reflection of water off the stone bank!",
            "medium": "Scanning the acoustic echoes of the wetland basin.",
            "low": "Sub-audible substrate vibrations.",
            "context": "Seismic and acoustic detection of wetland boundaries (Narins 1990)."
        }
    }
}


EMOTION_TRANSLATION_FLAVORS = {
    "Loneliness / Isolation": "I've been feeling so lonely by myself, and I really missed having you around.",
    "Stress — Chronic": "I've been feeling a bit stressed and anxious lately.",
    "Stress — Acute": "That startled me and I'm really on edge right now!",
    "Contentment": "I feel so peaceful, happy, and safe with you right now.",
    "Pain / Discomfort": "Something hurts or feels uncomfortable, so please be gentle with me.",
    "Social Bonding": "You are my favorite companion and I love being right here with you.",
    "Excitement / Anticipation": "I can barely contain my excitement! This is so thrilling!",
    "Fear / Distress": "I'm really frightened right now! Please stay close and protect me!",
    "Playfulness": "I'm bursting with energy and ready for a fun game!",
    "Vigilance / Alertness": "I'm on high alert watching everything very closely."
}


def generate_human_translation(
    species: str,
    behavior: str,
    intensity: float,
    emotion: Optional[str] = None,
    breed: Optional[str] = None,
    urgency: float = 0.5
) -> Dict[str, str]:
    """
    Generate authentic, natural first-person human language translation
    grounded in bioacoustic and ethological science.
    """
    species_key = (species or "dogs").lower().strip()
    if species_key not in ETHOLOGY_TRANSLATION_MAP:
        species_key = "dogs"

    behavior_key = (behavior or "contact_call").lower().strip()
    species_behaviors = ETHOLOGY_TRANSLATION_MAP[species_key]

    # Find matching behavior or fallback to contact_call or feeding_call
    if behavior_key not in species_behaviors:
        behavior_key = "feeding_call" if "feed" in behavior_key else "contact_call"

    entry = species_behaviors[behavior_key]

    # Select intensity level
    if intensity >= 0.7 or urgency >= 0.75:
        base_text = entry["high"]
    elif intensity >= 0.35:
        base_text = entry["medium"]
    else:
        base_text = entry["low"]

    # Blend emotional flavor if available and relevant
    if emotion and isinstance(emotion, str):
        for emo_key, flavor in EMOTION_TRANSLATION_FLAVORS.items():
            if emo_key.lower().split(" ")[0] in emotion.lower():
                # Blend with emotional context for high naturalness
                if "lonel" in emotion.lower() and behavior_key in ("feeding_call", "contact_call", "distress_call"):
                    base_text = f"Hey! {flavor} {base_text}"
                elif "stress" in emotion.lower() and behavior_key in ("distress_call", "alarm_ground", "aggression_warning"):
                    base_text = f"{base_text} {flavor}"
                elif "content" in emotion.lower() and behavior_key in ("social_bonding", "play_vocalization"):
                    base_text = f"{base_text} {flavor}"
                elif "pain" in emotion.lower() and behavior_key in ("distress_call", "submission_signal"):
                    base_text = f"{flavor} {base_text}"
                break

    return {
        "translation": base_text,
        "vocalization_type": entry["name"],
        "emotion": emotion or entry.get("emotion", "alert / expressive"),
        "behavior_context": entry["context"],
        "confidence": "high" if intensity > 0.6 else "medium"
    }
