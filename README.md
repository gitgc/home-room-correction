# roomeq

Room correction for stereo speakers from the command line. Put a calibrated
**miniDSP UMIK-1** at your listening position and run one command. roomeq
measures each speaker from several mic positions and fits a **10-band
parametric EQ** that brings the averaged in-room response onto a target
curve. It writes ready-to-load files for:

- **Equalizer APO** (Windows): per-channel EQ, level trim and delay
- **SoundSource** (macOS): AutoEq `ParametricEQ.txt` format
- **miniDSP 2x4 HD** (and other miniDSP plugins): 96 kHz biquad coefficients per output

Each run also writes a before/after chart:

![example report](docs/example-report.png)

After loading the files, `roomeq verify` re-measures with the correction
switched on. It checks the EQ is doing what was designed and pinpoints the
usual set-up mistakes (see [Verifying a correction](#verifying-a-correction)).

## How it works

The method follows what commercial room correction (Dirac Live, REW's
auto-EQ) does for the magnitude response:

1. **Log-sine sweeps** (Farina method), played on L, then R, then L again in
   one continuous recording. Deconvolution yields a clean impulse response for
   each speaker and moves harmonic distortion out of the analysis window. The
   repeated L sweep measures clock drift between the DAC and the mic, so the
   L/R arrival difference is accurate to a few microseconds.
2. **Multi-position spatial averaging.** As with Dirac, you measure several
   mic positions around your head (default 5, main seat weighted ×2). Their
   power average describes what you hear across the listening area, not one
   point in space. EQ fitted to a single point fixes nulls that move when you
   shift your head, and can make things worse.
3. **Mic calibration** applied from the `mic/` folder.
4. **Psychoacoustic smoothing.** 1/24 octave in the bass, where room modes
   live, widening to 1/3 octave from 1 kHz up, which is what the ear resolves
   tonally.
5. **Target curve.** A Harman/Dirac-style in-room curve by default: +4 dB
   bass shelf, about −0.8 dB/octave tilt above 200 Hz. A flat in-room
   response sounds thin and bright.
6. **EQ fitting** with room-correction rules:
   - Peaks are cut fully. Dips are only partly filled: boosts are limited by a
     1/3-octave-smoothed response and by a hard cap (+4 dB in the bass,
     +3 dB in the treble). Deep narrow nulls are cancellation and can't be
     fixed with EQ.
   - **No boosting beyond the speaker's own bandwidth.** Both bass and treble
     extension are detected automatically.
   - Precise, high-Q correction below the Schroeder frequency (≈300 Hz, the
     room-mode region). Above it, only broad gentle correction. Narrow cuts
     are allowed for resonances; narrow boosts never are.
   - Limits apply to the combined curve, so stacked filters can't get around them.
   - Greedy band selection plus joint non-linear least-squares refinement
     of all bands.
7. **L/R alignment.** Arrival-time difference becomes a delay on the nearer
   speaker, and the level difference becomes a trim on the louder one. A
   polarity mismatch (reversed speaker wire) is flagged.

### Compared with Dirac Live

roomeq aims to match Dirac's *frequency-response* correction. Dirac also does
mixed-phase *impulse response* correction with long FIR filters; a 10-band
minimum-phase IIR EQ can't do that, by design. In practice most of the audible
benefit at home comes from taming bass room modes, tonal balance and L/R
matching, and roomeq covers all three.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone <this repo> && cd home-room-correction
uv sync
uv run roomeq --help
```

Put your mic's calibration files in `mic/` (from miniDSP, keyed by the serial
number on the mic). The file whose name contains `90` is the 90° file; the
other one is the 0° file.

## Measuring

1. **Turn off existing correction** on the output you're measuring: Equalizer
   APO, SoundSource EQ, miniDSP PEQ/Dirac. Otherwise you measure your old EQ.
2. Plug in the UMIK-1. Mount it on a stand or tripod at **ear height,
   pointing straight up at the ceiling** (this uses the 90° calibration file;
   it's also how Dirac Live has you measure). Both speakers are then at the
   same angle to the mic, and you never re-aim between channels or positions.
3. Set the volume:

   ```sh
   uv run roomeq levels
   ```

   It plays pink noise on each speaker. Adjust your amp until it reports
   `good` (mic RMS around −40 to −20 dBFS, roughly 70–80 dB SPL at the seat).
4. Measure:

   ```sh
   uv run roomeq measure --name living-room
   ```

   It prompts you through 5 mic positions: the main seat, then about 30 cm
   left, right, forward and back. Use `--positions 9` for a more Dirac-like
   spread. Keep the room quiet; each position takes about 15 seconds.

The results go to `corrections/living-room_<date>_<time>/` and the chart
opens automatically.

5. Load the files into your EQ ([Output files](#output-files)), then check them
   with `uv run roomeq verify corrections/living-room_<date>_<time>`
   ([Verifying a correction](#verifying-a-correction)).

Useful options:

| option | default | |
| --- | --- | --- |
| `--output NAME` | system default | playback device (name substring or index from `roomeq devices`) |
| `--channels 1,2` | `1,2` | output channels used for left,right |
| `--positions N` | 5 | mic positions |
| `--target` | `default` | `default`, `warm`, `neutral`, `flat`, or a REW-style house-curve file |
| `--bass-db`, `--tilt` | preset | adjust the target's bass shelf (dB) / tilt (dB per octave) |
| `--max-boost` | 4 | max boost, dB |
| `--max-freq` | 16000 | highest corrected frequency (e.g. `500` for bass-only correction) |
| `--min-freq` | auto | lowest corrected frequency |
| `--filters` | 10 | number of bands |
| `--mic-orientation` | 90 | `0` if the mic points at the speakers instead |

### Re-computing without re-measuring

Every correction folder keeps the raw impulse responses
(`measurement.npz`). You can try a different target or limits without
measuring again:

```sh
uv run roomeq compute corrections/living-room_2026-11-10_1530 --target warm --open
uv run roomeq compute corrections/living-room_2026-11-10_1530 --max-freq 500 --out /tmp/bass-only
```

## Output files

| file | for |
| --- | --- |
| `EqualizerAPO.txt` | Equalizer APO. Copy it to `C:\Program Files\EqualizerAPO\config\` and add `Include: EqualizerAPO.txt` to `config.txt`. Contains per-channel preamp, delay and filters. |
| `Combined_ParametricEQ.txt` | SoundSource. One EQ for both channels (fitted to the L+R average), in AutoEq format; import it as a custom parametric EQ preset. |
| `Left_ParametricEQ.txt`, `Right_ParametricEQ.txt` | Per-channel EQ in AutoEq format (for anything that takes them, or for manual entry). |
| `miniDSP_Left_biquads.txt`, `miniDSP_Right_biquads.txt` | miniDSP 2x4 HD: Output 1/2 → PEQ → Advanced → Import. Coefficients are for 96 kHz in miniDSP's sign convention. Set input gain to the headroom and the output gains/delays from `summary.txt`. |
| `report.png` | Before/after chart for each channel and the stereo EQ. |
| `summary.txt` / `summary.json` | Filters, gains, delays, speaker distance difference, error before/after, warnings. |
| `measurement.npz` | The measured impulse responses and measurement details, used by `roomeq compute` and `roomeq verify`. Raw mic audio isn't kept. |
| `verify_*/` | One folder per `roomeq verify` run (see below). |

Filter format (all apps):

```text
Preamp: -4.0 dB
Filter 1: ON PK Fc 146.2 Hz Gain -12.9 dB Q 7.94
Filter 2: ON PK Fc 190.8 Hz Gain -9.6 dB Q 7.37
...
```

## Verifying a correction

Once the correction is loaded, check it actually works:

```sh
uv run roomeq verify corrections/living-room_2026-11-10_1530                # Equalizer APO / miniDSP
uv run roomeq verify corrections/living-room_2026-11-10_1530 --eq combined  # SoundSource
```

Switch on only the correction you're checking, keep the volume where it was if
you can, and put the mic in the same positions as the original measurement. By
default verify uses the same number of positions. `--positions 1` (main seat
only) is a valid quick check: verify compares only the positions it
re-measured.

**How it decides.** Verify doesn't compare the new response against the
target. Mic placement and the room would muddy that. It compares the
*change* since the original measurement with what the EQ should have done: it
applies the designed filters to the original measurement, smooths both the
same way, and scores the difference as dB RMS over the corrected range.

| result | measured vs expected | meaning |
| --- | --- | --- |
| PASS | under 1.5 dB | the correction is working as designed |
| WARN | 1.5–3 dB | usually mic placement; re-run using the same positions |
| FAIL | over 3 dB, or a specific problem found | see the diagnosis |

It also tests the usual set-up mistakes as alternative explanations and
reports the one that fits:

| diagnosis | typical cause |
| --- | --- |
| EQ is not active | Equalizer APO not enabled for that device, SoundSource effect bypassed, wrong miniDSP preset |
| Filters at half/double their frequency | biquad coefficients made for a different sample rate than the device runs at |
| EQ is inverted | gains entered with the wrong sign |
| Left and right swapped | Left file loaded on the right output |
| Level balance off | channel trims / output gains not applied |
| Delay not applied / out of alignment | speaker delay missing or wrong (or the mic moved) |
| Extreme deviations | an unstable filter, e.g. a miniDSP biquad import gone wrong |

It also reports the overall level change against the expected preamp. That's
informational, since it only holds if the volume wasn't touched.

Verify only measures. It never changes Equalizer APO, SoundSource or the
miniDSP, and never overwrites the correction. Each run writes
`verify_<channel|combined>_<date>_<time>/` inside the correction folder,
containing `verify.txt`, `verify.json`, `verify_report.png` and that run's
`measurement.npz`. The chart opens automatically (`--no-open` to skip). It
shows before, predicted and measured responses for each channel, and the
expected vs measured EQ curves:

![example verify report - simulated wrong sample rate](docs/example-verify-fail.png)

*(Simulated example: miniDSP coefficients made for 96 kHz running at 48 kHz.
The measured EQ, solid green, sits an octave below the expected one, dashed.)*

The exit code is 0 for PASS/WARN and 2 for FAIL, so it can be scripted.

## Development

```sh
uv run pytest
uv run roomeq measure --simulate --yes --positions 3 --out /tmp/sim   # synthetic room, no hardware
uv run roomeq verify /tmp/sim --simulate --yes                          # EQ applied correctly
uv run roomeq verify /tmp/sim --simulate --simulate-eq swapped --yes    # or: none, inverted, halfrate
```

Code layout (`roomeq/`): `sweep.py` (stimulus, deconvolution, timing),
`analysis.py` (windowing, smoothing, spatial average), `target.py`,
`optimizer.py` (EQ fitting), `peq.py` (biquad maths), `export.py`,
`report.py`, `audio.py` (device I/O plus simulated room), `pipeline.py`,
`verify.py` (re-measurement comparison and diagnosis), `cli.py`.

On macOS, the first run needs microphone permission for your terminal app
(System Settings → Privacy & Security → Microphone).

## License

MIT
