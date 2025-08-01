(
s.options.memSize = 65536; // Increase real-time memory to 64 MB
s.waitForBoot {
    var toxicityBus, activeSynths;

    // Control bus for toxicity score
    toxicityBus = Bus.control(s, 1).set(0.0); // Initialize to 0

    // Track active synths per toxicity level
    activeSynths = Dictionary.newFrom([\Mild, 0, \Moderate, 0, \High, 0]);

    // Define Synth for background pad (eery, dark, texturally rich)
    SynthDef(\darkPad, {
        |amp = 0.1, toxicityBus|
        var sig, freqMod, noise, filterMod, highPitch, toxicity, formant, feedback;
        toxicity = In.kr(toxicityBus).lag(0.5);
        freqMod = SinOsc.kr(0.1, 0, 0.02);
        // Layered low-frequency oscillators
        sig = SinOsc.ar(60 * (1 + freqMod)) * 0.4;
        sig = sig + Pulse.ar(59 * (1 + freqMod), 0.4, 0.3);
        sig = sig + SinOsc.ar(61 * (1 + freqMod), 0, 0.2); // Microtonal detuning
        // High-pitched glitchy elements
        highPitch = Blip.ar(1000 * (1 + freqMod), 4 + (toxicity * 4)) * 0.1 * toxicity;
        highPitch = highPitch * SinOsc.ar(1000 * (1 + LFNoise1.kr(0.3, 0.02)), 0, 0.5, 0.5);
        sig = sig + highPitch;
        // Textured noise with formant
        noise = PinkNoise.ar(0.05 + (toxicity * 0.1));
        noise = noise + WhiteNoise.ar(0.02 * toxicity); // Subtle white noise
        formant = Formlet.ar(noise, 200 + (toxicity * 300), 0.01, 0.02) * 0.5;
        noise = LPF.ar(noise + formant, 200 + SinOsc.kr(0.2, 0, 100));
        sig = sig + noise;
        // Subtle feedback for richness
        feedback = LocalIn.ar(1);
        sig = sig + (feedback * 0.2);
        LocalOut.ar(DelayC.ar(sig, 0.5, 0.3 * toxicity));
        // Modulated filter
        filterMod = 800 + SinOsc.kr(0.15, 0, 400);
        sig = LPF.ar(sig, filterMod * (1 - (toxicity * 0.5)));
        sig = (sig * (1.2 + (toxicity * 0.5))).tanh;
        sig = FreeVerb.ar(sig, 0.8, 0.9, 0.3);
        sig = Pan2.ar(sig, SinOsc.kr(0.05));
        sig = sig * amp * (1 + (toxicity * 0.2));
        sig = sig * SinOsc.kr(0.1, 0, 0.2, 0.8);
        Out.ar(0, sig);
    }).add;
    ("SynthDef \\darkPad added").postln;

    // Define Synth for mild toxicity (glitchy, eery, subtle IDM textures)
    SynthDef(\mildToxicSynth, {
        |freq = 1000, amp = 0.3, dur = 5, toxicity = 0.4|
        var sig, freqMod, noise, env, stutter, grains, feedback;
        freqMod = LFNoise1.kr(0.5, 0.05); // Erratic modulation
        sig = Saw.ar(freq * (1 + freqMod)) * 0.3;
        sig = sig + Pulse.ar(freq * 0.99 * (1 + freqMod), 0.4, 0.2);
        grains = TGrains.ar(2, Dust.ar(20 * toxicity), Buffer.alloc(s, 0.1 * s.sampleRate, 1),
                            Saw.ar(freq * 0.5), 0, 0.05, 0.1, pan: LFNoise1.kr(0.5));
        sig = sig + (grains * 0.3);
        feedback = LocalIn.ar(1);
        sig = sig + (feedback * 0.3);
        LocalOut.ar(DelayC.ar(sig, 0.2, 0.1 * toxicity));
        stutter = sig * LFPulse.kr(4 + (toxicity * 4), 0, 0.3);
        sig = sig + (stutter * 0.4);
        sig = Decimator.ar(sig, 44100 * (1 - toxicity * 0.5), 16);
        noise = PinkNoise.ar(0.05 * toxicity) * Dust.kr(10 * toxicity);
        noise = LPF.ar(noise, 800 + (toxicity * 500));
        sig = sig + noise;
        sig = LPF.ar(sig, 1500 + (toxicity * 1000));
        sig = (sig * (1 + (toxicity * 0.5))).tanh;
        sig = FreeVerb.ar(sig, 0.6, 0.8, 0.4);
        sig = Pan2.ar(sig, LFNoise1.kr(0.5));
        env = EnvGen.kr(Env.perc(0.05, dur - 0.05), doneAction: 2);
        sig = sig * env * amp * 2;
        Out.ar(0, sig);
    }).add;
    ("SynthDef \\mildToxicSynth added").postln;

    // Define Synth for moderate toxicity (glitchy, eery, pronounced IDM textures)
    SynthDef(\moderateToxicSynth, {
        |freq = 1500, amp = 0.4, dur = 5.5, toxicity = 0.6|
        var sig, freqMod, noise, stutter, grains, feedback, env;
        freqMod = LFNoise0.kr(1, 0.06);
        sig = Pulse.ar(freq * (1 + freqMod), 0.5) * 0.3;
        sig = sig + Blip.ar(freq * 0.98 * (1 + freqMod), 6 + (toxicity * 8), 0.3);
        sig = Decimator.ar(sig, 44100 * (1 - toxicity * 0.7), 12);
        grains = TGrains.ar(2, Dust.ar(30 * toxicity), Buffer.alloc(s, 0.1 * s.sampleRate, 1),
                            Pulse.ar(freq * 0.7), 0, 0.03, 0.15, pan: LFNoise1.kr(0.7));
        sig = sig + (grains * 0.4);
        feedback = LocalIn.ar(1);
        sig = sig + (feedback * 0.5);
        LocalOut.ar(DelayC.ar(sig, 0.3, 0.15 * toxicity));
        stutter = sig * LFPulse.kr(8 + (toxicity * 8), 0, 0.2);
        sig = sig + (stutter * 0.5);
        sig = FreqShift.ar(sig, LFNoise1.kr(0.5, 100 * toxicity));
        noise = PinkNoise.ar(0.1 * toxicity) * Dust.kr(15 * toxicity);
        noise = LPF.ar(noise, 1000 + (toxicity * 600));
        sig = sig + noise;
        sig = LPF.ar(sig, 2000 + (toxicity * 1200));
        sig = (sig * (1.5 + (toxicity * 1))).tanh;
        sig = FreeVerb.ar(sig, 0.7, 0.85, 0.3);
        sig = Pan2.ar(sig, LFNoise1.kr(0.6));
        env = EnvGen.kr(Env.perc(0.05, dur - 0.05), doneAction: 2);
        sig = sig * env * amp * 2;
        Out.ar(0, sig);
    }).add;
    ("SynthDef \\moderateToxicSynth added").postln;

    // Define Synth for high toxicity (glitchy, eery, chaotic IDM textures)
    SynthDef(\highToxicSynth, {
        |freq = 2000, amp = 0.5, dur = 6, toxicity = 0.8|
        var sig, fm, noise, feedback, env, glitch, sampleHold, grains;
        sig = Pulse.ar(freq * (1 + SinOsc.kr(0.3, 0, 0.07)), 0.5) * 0.3;
        sig = sig + Blip.ar(freq * 0.99, 10 + (toxicity * 10), 0.3);
        sig = Decimator.ar(sig, 44100 * (1 - toxicity * 0.9), 8);
        fm = SinOsc.ar(freq * (1 + SinOsc.kr(0.4, 0, 0.1)), SinOsc.ar(freq * 2) * 0.5) * 0.3;
        sig = sig + fm;
        feedback = LocalIn.ar(1);
        sig = sig + (feedback * 0.7);
        LocalOut.ar(DelayC.ar(sig, 0.4, 0.2 * toxicity));
        sampleHold = Latch.ar(sig, Dust.kr(20 * toxicity));
        sig = SelectX.ar(LFNoise1.kr(0.5).range(0, 1), [sig, sampleHold]);
        glitch = sig * LFPulse.kr(12 + (toxicity * 12), 0, 0.15);
        sig = sig + (glitch * 0.6);
        grains = TGrains.ar(2, Dust.ar(40 * toxicity), Buffer.alloc(s, 0.1 * s.sampleRate, 1),
                            Blip.ar(freq * 0.8), 0, 0.02, 0.2, pan: LFNoise1.kr(0.8));
        sig = sig + (grains * 0.5);
        noise = PinkNoise.ar(0.15 * toxicity) * Dust.kr(20 * toxicity);
        noise = LPF.ar(noise, 1200 + (toxicity * 800));
        sig = sig + noise;
        sig = FreqShift.ar(sig, LFNoise1.kr(0.7, 150 * toxicity));
        sig = (sig * (2 + (toxicity * 2))).tanh;
        sig = LPF.ar(sig, 2500 * (1 - (toxicity * 0.5)));
        sig = FreeVerb.ar(sig, 0.8, 0.9, 0.2);
        sig = Pan2.ar(sig, LFNoise1.kr(0.5));
        env = EnvGen.kr(Env.perc(0.05, dur - 0.05), doneAction: 2);
        sig = sig * env * amp * 2;
        Out.ar(0, sig);
    }).add;
    ("SynthDef \\highToxicSynth added").postln;

    // Start background pad
    s.makeBundle(nil, {
        Synth(\darkPad, [\amp, 0.1, \toxicityBus, toxicityBus]);
    });
    ("Started \\darkPad").postln;

    // Setup OSC listeners
    [
        ["/mildToxic", \mildToxicSynth, "Mild"],
        ["/moderateToxic", \moderateToxicSynth, "Moderate"],
        ["/highToxic", \highToxicSynth, "High"]
    ].do { |route|
        var addr = route[0], synthName = route[1], level = route[2];
        OSCdef(("toxic" ++ level).asSymbol, { |msg|
            var toxicity, tweetText, duration, synth, baseFreq, amp, dur;
            toxicity = msg[1]; // Score (float)
            tweetText = msg[2].asString; // Tweet text (string)
            duration = msg[3] ? switch(level, // Default durations
                "Mild", { 5.0 },
                "Moderate", { 5.5 },
                "High", { 6.0 }
            ); // Duration (float, seconds)
            ("Processing OSC: addr=" ++ addr ++ ", toxicity=" ++ toxicity ++ ", duration=" ++ duration ++ ", level=" ++ level).postln;

            // Limit concurrent synths (max 10 per level)
            if (activeSynths[level.asSymbol] < 10) {
                // Calculate frequency and duration
                baseFreq = switch(level,
                    "Mild", { toxicity * 2000 + 1000 }, // 1000–2000 Hz
                    "Moderate", { toxicity * 2000 + 1500 }, // 1500–2500 Hz
                    "High", { toxicity * 3000 + 2000 } // 2000–4000 Hz
                ).clip(1000, 4000);
                amp = switch(level,
                    "Mild", { 0.3 },
                    "Moderate", { 0.4 },
                    "High", { 0.5 }
                );
                dur = duration.clip(0.5, 10.0); // Safety: 0.5s–10s
                ("Computed: freq=" ++ baseFreq ++ ", amp=" ++ amp ++ ", dur=" ++ dur).postln;

                // Increment active synth count
                activeSynths[level.asSymbol] = activeSynths[level.asSymbol] + 1;

                // Play synth after darkPad
                s.makeBundle(nil, {
                    synth = Synth(synthName, [\freq, baseFreq, \amp, amp, \dur, dur, \toxicity, toxicity], addAction: \addToTail);
                    NodeWatcher.register(synth);
                    ("Created " ++ level ++ " synth (node " ++ synth.nodeID ++ ") with duration=" ++ dur ++ ". Active: " ++ activeSynths[level.asSymbol]).postln;
                    s.queryAllNodes;
                });

                // Handle synth release
                synth.onFree({
                    activeSynths[level.asSymbol] = activeSynths[level.asSymbol] - 1;
                    ("Freed " ++ level ++ " synth (node " ++ synth.nodeID ++ ") after duration=" ++ dur ++ ". Active: " ++ activeSynths[level.asSymbol]).postln;
                });

                // Force free synth after duration + buffer
                Routine({
                    (dur + 0.2).wait;
                    if (synth.isPlaying) {
                        ("Force freeing stuck " ++ level ++ " synth (node " ++ synth.nodeID ++ ") after duration=" ++ dur).postln;
                        synth.free;
                        activeSynths[level.asSymbol] = activeSynths[level.asSymbol] - 1;
                    };
                }).play;

                // Log received message
                ("Received " ++ level ++ " toxic tweet: Score = " ++ toxicity ++ ", Text = " ++ tweetText ++ ", Duration = " ++ dur).postln;
            } {
                ("Skipped " ++ level ++ " synth: too many active (" ++ activeSynths[level.asSymbol] ++ ")").postln;
            };
        }, addr, nil, 57120);
    };

    // OSC listener for TTS trigger
    OSCdef(\ttsTrigger, { |msg|
        var toxicity = msg[1], duration = msg[2];
        ("TTS Trigger: toxicity=" ++ toxicity ++ ", duration=" ++ duration).postln;
    }, '/ttsTrigger', nil, 57120);
};
)