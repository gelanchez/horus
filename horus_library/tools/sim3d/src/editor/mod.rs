//! Scene editor and GUI for interactive simulation manipulation
//!
//! This module provides a visual-first editing experience for sim3d:
//! - **Spawn Palette**: Click buttons to add objects (no coding needed)
//! - **Property Editor**: Sliders to edit mass, friction, color
//! - **Robot Browser**: Visual URDF picker with presets
//! - **Scene Manager**: Save/load scenes (auto-generates YAML)
//!
//! The editor is enabled with the "visual" feature for basic UI,
//! or "editor" feature for advanced tools.

// Core editor modules (require visual feature for egui)
#[cfg(feature = "visual")]
pub mod camera;
#[cfg(feature = "visual")]
pub mod gizmos;
#[cfg(feature = "visual")]
pub mod hierarchy;
#[cfg(feature = "visual")]
pub mod inspector;
#[cfg(feature = "visual")]
pub mod selection;
#[cfg(feature = "visual")]
pub mod ui;
#[cfg(feature = "visual")]
pub mod undo;

// Visual-first UX modules (new)
#[cfg(feature = "visual")]
pub mod property_editor;
#[cfg(feature = "visual")]
pub mod robot_browser;
#[cfg(feature = "visual")]
pub mod scene_manager;
#[cfg(feature = "visual")]
pub mod spawn_palette;
#[cfg(feature = "visual")]
pub mod visual_editor;

#[cfg(feature = "visual")]
use bevy::prelude::*;

#[cfg(feature = "visual")]
use bevy_egui::EguiPlugin;

/// Editor state and configuration
#[cfg(feature = "visual")]
#[derive(Resource, Default)]
pub struct EditorState {
    /// Whether the editor is enabled
    pub enabled: bool,
    /// Show inspector panel
    pub show_inspector: bool,
    /// Show hierarchy panel
    pub show_hierarchy: bool,
    /// Show toolbar
    pub show_toolbar: bool,
    /// Grid snapping enabled
    pub snap_to_grid: bool,
    /// Grid size for snapping
    pub grid_size: f32,
    /// Current gizmo mode
    pub gizmo_mode: GizmoMode,
    /// Editor camera mode
    pub camera_mode: EditorCameraMode,
}

#[cfg(feature = "visual")]
impl EditorState {
    pub fn new() -> Self {
        Self {
            enabled: true,
            show_inspector: true,
            show_hierarchy: true,
            show_toolbar: true,
            snap_to_grid: false,
            grid_size: 0.1,
            gizmo_mode: GizmoMode::Translate,
            camera_mode: EditorCameraMode::Orbit,
        }
    }
}

/// Gizmo manipulation mode
#[cfg(feature = "visual")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
    None,
}

/// Editor camera control mode
#[cfg(feature = "visual")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
pub enum EditorCameraMode {
    #[default]
    Orbit,
    Pan,
    Fly,
    Top,
    Side,
    Front,
}

/// Main editor plugin - provides the visual-first UX for sim3d
///
/// This plugin enables:
/// - Click-to-spawn objects from the palette
/// - Drag-to-move with transform gizmos
/// - Sliders to edit physics properties
/// - Visual robot browser with presets
/// - Auto-save/load scenes (generates YAML automatically)
#[cfg(feature = "visual")]
pub struct EditorPlugin;

#[cfg(feature = "visual")]
impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin);
        }
        app
            // Core editor state
            .init_resource::<EditorState>()
            .init_resource::<selection::Selection>()
            .init_resource::<undo::UndoStack>()
            .init_resource::<hierarchy::HierarchyCollapseState>()
            // New visual UX resources
            .init_resource::<spawn_palette::SpawnPaletteState>()
            .init_resource::<scene_manager::SceneManagerState>()
            .init_resource::<robot_browser::RobotBrowserState>()
            // Events for visual workflow
            .add_event::<spawn_palette::SpawnObjectEvent>()
            .add_event::<spawn_palette::SpawnRobotEvent>()
            .add_event::<scene_manager::SaveSceneEvent>()
            .add_event::<scene_manager::LoadSceneEvent>()
            .add_event::<scene_manager::ClearSceneEvent>()
            .add_event::<robot_browser::LoadRobotEvent>()
            .add_event::<hierarchy::DeleteEntityEvent>()
            .add_event::<hierarchy::DuplicateEntityEvent>()
            .add_event::<undo::UndoEvent>()
            .add_event::<undo::RedoEvent>()
            // Core editor systems
            .add_systems(
                Update,
                (
                    ui::editor_ui_system,
                    hierarchy::hierarchy_panel_system,
                    gizmos::gizmo_system,
                )
                    .run_if(editor_enabled),
            )
            .add_systems(
                Update,
                (
                    selection::selection_system,
                    camera::editor_camera_system,
                    undo::undo_keyboard_system,
                    hierarchy::delete_entity_system,
                    hierarchy::duplicate_entity_system,
                )
                    .run_if(editor_enabled),
            )
            .add_systems(
                Last,
                (undo::process_undo_system, undo::process_redo_system).chain(),
            )
            // Visual UX systems (the new stuff!)
            .add_systems(
                Update,
                (
                    // Menu bar with File > New/Open/Save
                    scene_manager::menu_bar_system,
                    // Spawn palette at bottom
                    spawn_palette::spawn_palette_panel_system,
                    // Quick spawn menu (Space key)
                    spawn_palette::quick_spawn_menu_system,
                    // Process spawn events
                    spawn_palette::process_spawn_events_system,
                    // Robot browser window
                    robot_browser::robot_browser_system,
                    robot_browser::process_load_robot_system,
                    // Scene save/load
                    scene_manager::save_scene_system,
                    scene_manager::load_scene_system,
                    scene_manager::clear_scene_system,
                    // Property editor panel (replaces inspector)
                    property_editor::property_editor_panel_system,
                )
                    .run_if(editor_enabled),
            )
            .register_type::<selection::Selectable>()
            .register_type::<EditorCameraMode>();
    }
}

#[cfg(feature = "visual")]
fn editor_enabled(state: Res<EditorState>) -> bool {
    state.enabled
}

// Non-visual stub implementations
#[cfg(not(feature = "visual"))]
pub struct EditorPlugin;

#[cfg(not(feature = "visual"))]
impl bevy::app::Plugin for EditorPlugin {
    fn build(&self, _app: &mut bevy::app::App) {
        // No-op when visual feature is disabled (headless mode)
    }
}

// Re-export commonly used types
