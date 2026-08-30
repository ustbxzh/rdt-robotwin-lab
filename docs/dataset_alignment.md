# Dataset and language alignment

Native collection writes XPolicyLab trajectory format v1.0 directly:

```text
/state/{left_arm_joint_states,left_ee_joint_states,...}
/action/{left_arm_joint_states,left_ee_joint_states,...}
/vision/{cam_head,cam_left_wrist,cam_right_wrist}/colors
/instructions
/additional_info/frequency
```

For a captured sequence `q[0..N-1]`, state is `q[:-1]` and action is
`q[1:]`. RGB and other state observations also drop the final frame, so all
modalities have horizon `N-1`.

Language embeddings mirror the episode path under `lang_embeds/`:

```text
data/<dataset>/<task>/<env>/data/episode_0000007.hdf5
lang_embeds/<dataset>/<task>/<env>/data/episode_0000007.pt
```

The encoder prefers an already selected `/instruction`. For native collection,
it parses `/instructions` and selects `episode_index % instruction_count`.
The loader derives the exact same `.pt` path, ensuring image, state, action,
and language remain aligned at episode level.
