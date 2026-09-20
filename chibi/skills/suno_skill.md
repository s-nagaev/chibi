# Suno AI Music Generation Skill Guide

Based on "The Complete Guide to Mastering Suno" and the Suno Song Creator workflow.

## 1. Core Philosophy
- **Systematic Approach**: Move from guesswork to a research-driven process.
- **Structured Prompting**: Use specific, descriptive tags rather than abstract feelings.
- **Model Selection**: Choose the model based on the goal (fidelity vs. creativity).

## 2. Prompt Structure (1000 char limit)
Use the colon-and-quotes format for maximum clarity and separation of concerns.

```text
genre: "primary genre, sub-genre, era, specific style descriptors"
vocal: "gender, vocal type (e.g., alto, tenor), delivery style (e.g., intimate, belting, whisper)"
instrumentation: "key instruments, playing style (e.g., fingerpicked, distorted, synth-heavy)"
production: "sound quality, atmosphere, technical descriptors (e.g., lo-fi, wide stereo, tape warmth)"
mood: "emotional tone, energy level"
```

**Example:**
```text
genre: "indie folk rock, 2020s bedroom pop aesthetic"
vocal: "soft female alto, intimate delivery, whisper-to-belt dynamic"
instrumentation: "fingerpicked acoustic guitar, warm bass, minimal percussion"
production: "lo-fi intimacy, proximity effect, room tone"
mood: "melancholic, nostalgic, vulnerable"
```

## 3. Model Selection Guide

| Model | Best For | Strengths | Limitations |
|---|---|---|---|
| **v5** | Acoustic, Pop, Vocals-first | Cleanest audio, most natural vocals, high fidelity | Can be conservative; sometimes adds intro vocals unnecessarily |
| **v4.5** | Heavy genres (Rock, Metal), Reliable structure | Workhorse, consistent output, follows structure well | May mangle lyrics occasionally compared to v5 |
| **v4.5+** | Creative experimentation | Interesting surprises, unique fusions | Unstable, random elements |
| **v4** | Intentional chaos, creative spark | Unpredictable brilliance, "happy accidents" | Poor instruction adherence, outdated sound quality |

## 4. Lyric Writing & Meta Tags
Structure your lyrics to guide the AI's arrangement.

### Structure
- **[Verse]**: Storytelling, lower energy.
- **[Chorus]**: Main theme, higher energy, hook.
- **[Bridge]**: Variation, building tension.
- **[Outro]**: Fading out, conclusion.

### Advanced Meta Tags
- **Style Injection**: `[Verse 1 | intimate delivery | sparse instrumentation]`
- **Instrumental Breaks**: `[Instrumental Solo]`, `[Bass Drop]`, `[Guitar Solo]`
- **Vocal Cues**: `[Whisper]`, `[Shout]`, `[Choir]`, `[Spoken Word]`

### "AI Slop" Avoidance
Avoid clichés that flag lyrics as AI-generated.
- **Avoid**: "neon", "static", "wire", "circuits", "echoes", "shadows", "void", "broken", "ghost in the machine".
- **Use instead**: Concrete imagery, specific physical details, human-centered emotions, conversational language.

## 5. Genre-Specific Strategies

### Acoustic / Folk
- **Keywords**: `Proximity effect`, `Room tone`, `Single-mic capture`, `Natural dynamics`.
- **Tip**: Use v5 for the best vocal nuance and guitar realism.

### Electronic / EDM
- **Keywords**: `Synthesis control`, `Anti-sawtooth strategies` (for cleaner sounds), `Sidechain compression`.
- **Tip**: Specify the decade (e.g., "80s synth vibes") to ground the sound.

### Rock / Alternative
- **Keywords**: `Raw production`, `90s aesthetics`, `Distortion`, `Amp noise`.
- **Tip**: Rock has a "gravity well" towards Pop. Use specific sub-genres (e.g., "Post-Hardcore", "Shoegaze") to escape generic sounds.

### Pop
- **Keywords**: `Modern polish`, `Radio-ready`, `Vocal production`, `Autotune` (if desired).
- **Tip**: Focus on "hooky" melodies and clear structure.

## 6. Advanced Concepts

### Genre Clouds & Gravity Wells
- Suno links related genres (e.g., Rock is strongly linked to Pop).
- **Strategy**: To get a pure genre sound (e.g., pure Metal without Pop elements), you may need to explicitly exclude pop-related terms or use very specific sub-genre tags that are further from the "center" of the cloud.

### Copyright Safety
- **Do NOT use**: Artist names (Taylor Swift), Band names (The Beatles), Producer names, Song titles.
- **Use**: Genre + Era + Descriptors + Mood + Technical terms.
    - *Bad*: "Like Phoebe Bridgers"
    - *Good*: "Indie folk, female alto, confessional lyrics, 2020s bedroom pop production"

## 7. Workflow Checklist
1.  **Define Vision**: What is the mood, genre, and topic?
2.  **Research**: (Optional) Look up BPM, instruments, and song structures of similar real tracks.
3.  **Select Model**: v5 for polish, v4.5 for grit.
4.  **Draft Prompt**: Fill out the structured fields.
5.  **Write Lyrics**: Create original lyrics with meta tags (or use instrumental).
6.  **Generate & Iterate**: Don't expect perfection on try #1. Tweak one variable at a time.
