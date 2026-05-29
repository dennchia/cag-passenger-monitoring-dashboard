# CAG Dashboard Style Guide

## Direction

The V1 dashboard should use a dark operational monitoring style:

- dark slate background
- high-contrast live video panel
- compact right sidebar
- compact camera switcher inside the video panel when multiple cameras exist
- separate Passenger Assistance tab for filtered age/gender observation cards
- color only for status, alert severity, and connection state
- no landing-page or marketing layout

## Layout

Primary hierarchy:

1. Header with dashboard name and overall camera connection state.
2. Tab navigation for Operations and Passenger Assistance.
3. Operations tab: large live MJPEG video panel plus compact status sidebar.
4. Assistance tab: filters first, then compact person crop cards.

## Component Rules

- `VideoPlayer` owns the selected camera stream display and the camera selector.
- `SystemStatus` shows backend, selected camera health, and all-camera online count.
- `MetricsPanel` stays compact with latest 10 rows.
- `AlertsPanel` stays compact with latest 5 alerts and severity indicators.
- `AssistanceView` displays model-provided demographic observations as an assistance filter, not identification.
- Offline video state must follow the selected camera status from `/api/cameras`, not image error events.

## Update Rule

Update this file whenever dashboard visual direction, first-screen hierarchy, or major UI component responsibilities change.
