# Restricted-Zone Event Dashboard

The IntrudeAware dashboard records transitions between outside and inside the configured restricted zone.

## Event fields

- **Video Time**: timestamp relative to the source video, formatted as `HH:MM:SS.mmm`.
- **Frame**: zero-based source-frame index.
- **Track ID**: persistent ID assigned by the centroid tracker.
- **Status**: `ENTRY` or `EXIT`.
- **Event**: human-readable description.

## Duplicate-event prevention

An object produces one `ENTRY` event when it changes from outside to inside. Remaining inside does not create additional events. If the object changes from inside to outside, an `EXIT` event is emitted. Short tracker gaps are tolerated through the zone detector's missing-frame grace period.

## Video Overlay

The processed video now includes a translucent highlighted restricted zone, a labeled
`RESTRICTED ZONE` boundary, and a visible `ID <n>` label for every active track.
Active tracks currently inside the restricted zone are highlighted differently and are
labeled `ID <n> | IN ZONE`.
