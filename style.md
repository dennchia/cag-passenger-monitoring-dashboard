# CAG Dashboard Style Guide

## Style Direction

The dashboard should use a **Light Ops Board** style: clean, professional, operational, and easy to read during a CAG or school demo.

The first screen should feel like a monitoring dashboard for supervisors, not a marketing page, report page, or technical debug console. The main visual story is:

1. How many people are inside now.
2. How close the tent is to capacity.
3. Whether the system needs attention.
4. Which zones or cameras may be causing concern.

## Primary Focus

The visual center is **Count + Capacity**.

- The fused passenger count should be the largest element on the Operations Overview.
- Capacity percentage and capacity limit should sit close to the count, not far below the page.
- Normal, warning, critical, and offline states should be obvious through label, color, and placement.
- The alert banner should appear directly under the primary count/capacity area.

## Operations Overview Layout

Recommended first-screen hierarchy:

1. Header status row: dashboard title, run ID, last update, data age, and system status.
2. Primary command area: total fused count, capacity used, capacity bar, and capacity state.
3. Current alert strip: the most important active condition, using green, amber, red, or grey.
4. Monitoring area: tent zone floorplan as the main secondary visual, with compact system health and camera status nearby.
5. Supporting area: count-over-time chart and latest event summary.

The Operations Overview should answer these questions within a few seconds:

- Is the count normal, near capacity, or critical?
- Is the data live or stale?
- Are all cameras and fusion components online?
- Which tent zone is most crowded?
- Is there an active alert that needs attention?

## Enhanced Floorplan Direction

The reference dashboard image is useful mainly because it makes the floorplan feel like the visual anchor of the interface. For this project, borrow the **spatial dashboard** idea without copying the real-estate styling.

Current floorplan direction: **Enhanced Schematic**.

The CAG floorplan should feel more polished and map-like than a simple row of rectangles, but it should still be operational and easy to read. It should represent a monitored tent or temporary field setup, not a property interior or decorative architectural render.

The enhanced floorplan should include, where possible:

- tent boundary or monitored-area outline
- dynamic zone blocks using the existing `zones` data
- zone labels, counts, capacity, percentage used, and normal/warning/critical state
- entrance and exit markers to make passenger flow understandable
- camera position markers based on available camera data or safe static demo placement
- clear occupancy overlays using the same green, amber, red, and grey status language as the rest of the dashboard

The enhanced floorplan should not:

- use low-contrast beige/glass styling from the reference image
- rely on tiny labels or decorative microcharts
- look like a real-estate/property dashboard
- require exact tent geometry before the team has confirmed the real layout
- overpower the main Count + Capacity status area

If exact tent geometry, camera coordinates, or zone polygons become available later, the schematic can become more accurate. Until then, use a clean estimated layout that supports demo clarity.

## Component Rules

### Top Controls

- Keep configuration controls out of the main dashboard body.
- Use a compact top-right settings control for demo scenario, live JSON path, demo mode, auto-refresh, technical view, and capacity thresholds.
- Keep the theme toggle beside the settings control, not inside the operational content area.
- The first visible dashboard content should still be the header status row and Count + Capacity story.

### Header

- Keep the title short: `CAG Passenger Monitoring - Live Dashboard`.
- Show run ID, last update, data age, and system status in a compact row.
- Do not use large introductory text or project explanations on the dashboard screen.

### Count And Capacity

- Make the fused count dominant.
- Place capacity percentage and capacity limit beside it.
- Use a horizontal capacity bar for quick scanning.
- Use consistent state colors:
  - normal: green
  - warning: amber
  - critical: red
  - offline or stale: grey

### Alerts

- Show one current alert strip in the Operations Overview.
- Use plain operational wording.
- Keep detailed alert history in the technical or event-log area.

### Floorplan

- Treat the tent floorplan as the main spatial view.
- Keep zone labels, counts, capacity, and state readable.
- Use the same normal/warning/critical colors as the capacity status.
- Avoid decorative map styling that makes the counts harder to read.

### Live Stitched Feed

- Place the stitched camera feed in the right monitoring column beside the floorplan.
- Keep it above the compact camera status tiles so it acts as visual confirmation of the schematic map.
- The dashboard should embed a stream URL from the camera/stitching pipeline; it should not perform camera stitching itself.
- For the first live version, prefer an HTTP MJPEG/image stream such as `http://localhost:8080/stitched_feed`.
- If the feed is unavailable, show a clean placeholder and keep count, capacity, floorplan, and health visible.

### Camera Status

- Use **compact camera status tiles** until real processed snapshots are available.
- Each camera tile should show camera ID, online/offline state, count, FPS, and reliability notes if available.
- Keep per-camera tiles compact even when a stitched feed is available.

### System Health

- Keep system health visible but secondary to count/capacity.
- Include cameras, fusion engine, dashboard, power, network if available, and last data state.
- Use warnings only when they help explain reliability or data freshness.

### Charts And Logs

- Count-over-time should support the operational story, not dominate the page.
- Logs, raw JSON, schema tables, and detailed debug output belong in Technical View.

### Registered Persons

- Keep Registered Persons in a separate tab.
- Treat it as optional, consented, demo-only, and not safety-critical.
- Do not let registered-person status compete with the main passenger count.

## Visual Tone

- Use a light background with restrained neutral surfaces.
- Use color mainly for operational status, alerts, and zone risk.
- Prefer clear spacing, strong labels, and stable alignment over decorative effects.
- Avoid landing-page style sections, oversized explanations, and purely decorative graphics.
- The dashboard should feel credible for CAG evaluators and understandable to non-technical viewers.

### Theme Mode

The default dashboard mode should remain **Light Ops Board** because it is clearest for school and CAG demo settings.

The dashboard may include a compact top-right theme button that switches between light and dark mode. The dark mode should keep the same operational hierarchy, spacing, and status colors; it should not become a separate control-room redesign.

Theme rules:

- light mode is the default
- the theme button should be visually compact and button-like, similar in spirit to modern shadcn theme toggles
- dark mode must preserve readable tables, charts, floorplan labels, alerts, and status badges
- normal, warning, critical, and offline colors should remain consistent across both modes
- changing theme should not change dashboard data, selected scenario, thresholds, or operational status

## Update Rule

Update this file whenever there is a major style change, including:

- a different first-screen layout
- a different visual tone, such as dark mode or control-room style
- a change to the main visual priority
- a major change to camera, floorplan, alert, or system-health presentation
- a change in demo audience or presentation goal

When changing the dashboard UI, treat this file as the source of truth and keep `plan.md` focused on project scope, architecture, integration, and testing.
