//! HORUS State Machines Module
//!
//! Provides a robust hierarchical finite state machine (HFSM) implementation
//! for robot mode management and state-based control.
//!
//! # Key Concepts
//!
//! ## States and Transitions
//!
//! States represent discrete modes the system can be in. Transitions define
//! rules for moving between states, optionally triggered by events and
//! guarded by conditions.
//!
//! ```rust,ignore
//! use horus_core::state_machines::{StateMachine, State, Transition, Event};
//!
//! struct RobotContext {
//!     battery_level: f32,
//!     is_charging: bool,
//! }
//!
//! let mut fsm = StateMachine::<RobotContext>::new("power_management")
//!     .add_state(State::new("normal").initial())
//!     .add_state(State::new("low_battery"))
//!     .add_state(State::new("charging"))
//!     .add_transition(
//!         Transition::new("normal", "low_battery")
//!             .on_event("battery_low")
//!             .with_guard(|ctx| ctx.battery_level < 0.2)
//!     )
//!     .add_transition(
//!         Transition::new("low_battery", "charging")
//!             .on_event("dock_connected")
//!     )
//!     .build()?;
//! ```
//!
//! ## Entry/Exit/Tick Actions
//!
//! States support callbacks executed at different lifecycle points:
//!
//! - **on_entry**: Called when entering the state
//! - **on_exit**: Called when leaving the state
//! - **on_tick**: Called each update cycle while in the state
//!
//! ```rust,ignore
//! let state = State::<RobotContext>::new("moving")
//!     .on_entry(|ctx| {
//!         println!("Starting movement");
//!         ctx.motors_enabled = true;
//!     })
//!     .on_tick(|ctx| {
//!         ctx.update_position();
//!     })
//!     .on_exit(|ctx| {
//!         ctx.motors_enabled = false;
//!     });
//! ```
//!
//! ## Guard Conditions
//!
//! Guards are predicates that must be true for a transition to occur:
//!
//! ```rust,ignore
//! let transition = Transition::new("idle", "running")
//!     .on_event("start")
//!     .with_guard(|ctx| {
//!         ctx.battery_level > 0.2 && ctx.sensors_ok()
//!     });
//! ```
//!
//! ## Hierarchical States
//!
//! States can contain child states for complex mode hierarchies:
//!
//! ```rust,ignore
//! // Top-level state with children
//! let autonomous = State::new("autonomous")
//!     .with_child("navigating")
//!     .with_child("avoiding_obstacle");
//!
//! // Child states
//! let navigating = State::new("navigating")
//!     .with_parent("autonomous")
//!     .initial();  // Default child state
//! ```
//!
//! ## Event Processing
//!
//! Events can be processed immediately or queued:
//!
//! ```rust,ignore
//! // Immediate processing
//! fsm.process_event(&Event::new("start"), &mut ctx)?;
//!
//! // Queue for later
//! fsm.queue_event(Event::high_priority("emergency_stop"))?;
//! fsm.process_queue(&mut ctx)?;
//! ```
//!
//! ## Builder API
//!
//! Use the fluent builder for cleaner state machine construction:
//!
//! ```rust,ignore
//! use horus_core::state_machines::StateMachineBuilder;
//!
//! let fsm = StateMachineBuilder::<RobotContext>::new("robot_fsm")
//!     .initial_state("idle")
//!         .on_entry(|ctx| println!("Ready"))
//!         .done()
//!     .state("moving")
//!         .on_tick(|ctx| ctx.update())
//!         .with_timeout(Duration::from_secs(60), "timeout")
//!         .done()
//!     .transition("idle", "moving")
//!         .on_event("start")
//!         .with_guard(|ctx| ctx.ready())
//!         .done()
//!     .build()?;
//! ```
//!
//! ## Metrics and Monitoring
//!
//! State machines track metrics for monitoring:
//!
//! ```rust,ignore
//! let metrics = fsm.metrics();
//! println!("Transitions: {}", metrics.successful_transitions);
//! println!("Events processed: {}", metrics.events_processed);
//! ```
//!
//! # Factory Functions
//!
//! Common state machine patterns:
//!
//! ```rust,ignore
//! // Toggle between two states
//! let toggle = toggle_machine::<Context>("toggle", "off", "on", "toggle")?;
//!
//! // Sequential progression
//! let sequence = sequential_machine::<Context>(
//!     "sequence",
//!     &["step1", "step2", "step3"],
//!     "next"
//! )?;
//!
//! // Mode selection hub
//! let modes = mode_selector_machine::<Context>(
//!     "modes",
//!     &["manual", "auto", "calibration"],
//!     "select",
//!     "back"
//! )?;
//! ```

pub mod builder;
pub mod machine;
pub mod types;

// Re-export main types at module level
pub use builder::{
    mode_selector_machine, sequential_machine, toggle_machine, StateBuilder, StateMachineBuilder,
    TransitionBuilder,
};
pub use machine::{SharedStateMachine, StateMachine, TransitionRecord};
pub use types::{
    ActionFn, Event, EventPriority, GuardFn, State, StateConfig, StateId, StateMachineError,
    StateMachineMetrics, Transition, TransitionConfig, TransitionResult,
};

/// Prelude module for convenient imports.
///
/// ```rust,ignore
/// use horus_core::state_machines::prelude::*;
/// ```
pub mod prelude {
    pub use super::{
        Event, EventPriority, SharedStateMachine, State, StateId, StateMachine,
        StateMachineBuilder, StateMachineError, Transition, TransitionResult,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestContext {
        counter: i32,
    }

    #[test]
    fn test_module_exports() {
        // Verify that main types are accessible
        let _id = StateId::named("test");
        let _event = Event::new("test_event");
        let _priority = EventPriority::Normal;
    }

    #[test]
    fn test_simple_fsm() {
        let fsm = StateMachine::<TestContext>::new("test")
            .add_state(State::new("idle").initial())
            .add_state(State::new("running"))
            .add_transition(Transition::new("idle", "running").on_event("start"))
            .add_transition(Transition::new("running", "idle").on_event("stop"))
            .build()
            .unwrap();

        let mut ctx = TestContext { counter: 0 };
        let mut fsm = fsm;

        fsm.start(&mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("idle")));

        fsm.process_event(&Event::new("start"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("running")));

        fsm.process_event(&Event::new("stop"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("idle")));
    }

    #[test]
    fn test_builder_fsm() {
        let fsm = StateMachineBuilder::<TestContext>::new("builder_test")
            .initial_state("off")
            .on_entry(|ctx| ctx.counter = 0)
            .done()
            .state("on")
            .on_entry(|ctx| ctx.counter = 1)
            .done()
            .transition("off", "on")
            .on_event("toggle")
            .done()
            .transition("on", "off")
            .on_event("toggle")
            .done()
            .build()
            .unwrap();

        let mut ctx = TestContext { counter: -1 };
        let mut fsm = fsm;

        fsm.start(&mut ctx).unwrap();
        assert_eq!(ctx.counter, 0);

        fsm.process_event(&Event::new("toggle"), &mut ctx).unwrap();
        assert_eq!(ctx.counter, 1);

        fsm.process_event(&Event::new("toggle"), &mut ctx).unwrap();
        assert_eq!(ctx.counter, 0);
    }

    #[test]
    fn test_shared_fsm() {
        let fsm = StateMachine::<TestContext>::new("shared_test")
            .add_state(State::new("a").initial())
            .add_state(State::new("b"))
            .add_transition(Transition::new("a", "b").on_event("go"))
            .build()
            .unwrap();

        let shared = SharedStateMachine::new(fsm);
        let mut ctx = TestContext { counter: 0 };

        shared.start(&mut ctx).unwrap();
        assert!(shared.is_running());
        assert_eq!(shared.current_state(), Some(StateId::named("a")));

        shared.process_event(&Event::new("go"), &mut ctx).unwrap();
        assert_eq!(shared.current_state(), Some(StateId::named("b")));
    }

    #[test]
    fn test_factory_functions() {
        // Toggle machine
        let _toggle = toggle_machine::<TestContext>("toggle", "off", "on", "toggle").unwrap();

        // Sequential machine
        let _seq = sequential_machine::<TestContext>("seq", &["s1", "s2", "s3"], "next").unwrap();

        // Mode selector
        let _modes =
            mode_selector_machine::<TestContext>("modes", &["a", "b", "c"], "select", "back")
                .unwrap();
    }
}
