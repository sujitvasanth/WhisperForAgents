# Whisper for Agents

<table>
  <tr>
    <td width="50%" valign="top">

Real-time speech dictation inspired by <a href="https://github.com/KoljaB/RealtimeSTT#">RealtimeSTT</a>, rewritten to avoid slow, dependency-heavy initialization.

<h3>What’s different</h3>
<ul>
  <li><strong>Fast start-up:</strong> custom init path (no heavy library boot).</li>
  <li><strong>GPU-friendly models:</strong> distilled Whisper
    &rarr; <a href="https://huggingface.co/Systran/faster-distil-whisper-small.en">Systran/faster-distil-whisper-small.en</a>.
    On an RTX 3060, <strong>bf16</strong> + distill was ~&times;6 faster than baseline.</li>
  <li><strong>CPU fallback:</strong> works with <code>tiny.en</code> in <strong>int8</strong> on CPU.</li>
  <li><strong>Smart finalization:</strong> phrase confidence = lowest word confidence in the phrase; when a final is emitted, <strong>changes are shown in bold</strong>.</li>
  <li><strong>Noise suppression:</strong> basic denoise front-end improves SNR before VAD/ASR.</li>
</ul>

<h3>Why you might want this</h3>
<ul>
  <li>Lower latency from cold start.</li>
  <li>Stable streaming with clear interim → final transitions.</li>
  <li>More readable text in mildly noisy rooms.</li>
</ul>

</td>
    <td width="72%" valign="top" align="center">

<img src="https://github.com/user-attachments/assets/e33aa9b9-c6e7-4cb1-9028-b679ccc080fa"
     alt="Real-time dictation demo GIF"
     style="max-width:100%; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.08);" />

</td>
  </tr>
</table>

