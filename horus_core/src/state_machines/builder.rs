//! Builder pattern for constructing HORUS State Machines.
//!
//! This module provides a fluent API for building state machines with
//! compile-time safety and clear semantics.
//!
//! # Example
//!
//! ```rust,ignore
//! use horus_core::state_machines::StateMachineBuilder;
//!
//! struct RobotContext {
//!     battery: f32,
//!     position: (f64, f64),
//! }
//!
//! let fsm = StateMachineBuilder::<RobotContext>::new("navigation_fsm")
//!     .initial_state("idle")
//!         .on_entry(|ctx| println!("Entering idle"))
//!         .on_exit(|ctx| println!("Leaving idle"))
//!         .done()
//!     .state("moving")
//!         .on_entry(|ctx| println!("Starting movement"))
//!         .on_tick(|ctx| { /* update position */ })
//!         .with_timeout(Duration::from_secs(60), "timeout")
//!         .done()
//!     .state("charging")
//!         .final_state()
//!         .done()
//!     .transition("idle", "moving")
//!         .on_event("start")
//!         .with_guard(|ctx| ctx.battery > 0.2)
//!         .done()
//!     .transition("moving", "idle")
//!         .on_event("stop")
//!         .done()
//!     .build()?;
//! ```

use super::machine::StateMachine;
use super::types::*;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

/// Builder for constructing state machines with a fluent API.
pub struct StateMachineBuilder<C> {
    name: String,
    states: Vec<State<C>>,
    transitions: Vec<Transition<C>>,
    max_queue_size: usize,
    max_history_size: usize,
    _context: PhantomData<C>,
}

impl<C> StateMachineBuilder<C> {
    /// Create a new state machine builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            states: Vec::new(),
            transitions: Vec::new(),
            max_queue_size: 100,
            max_history_size: 50,
            _context: PhantomData,
        }
    }

    /// Add an initial state and return a state builder.
    pub fn initial_state(self, id: impl Into<StateId>) -> StateBuilder<C> {
        StateBuilder::new(self, id.into(), true)
    }

    /// Add a state and return a state builder.
    pub fn state(self, id: impl Into<StateId>) -> StateBuilder<C> {
        StateBuilder::new(self, id.into(), false)
    }

    /// Add a transition and return a transition builder.
    pub fn transition(
        self,
        from: impl Into<StateId>,
        to: impl Into<StateId>,
    ) -> TransitionBuilder<C> {
        TransitionBuilder::new(self, from.into(), to.into())
    }

    /// Set the maximum event queue size.
    pub fn with_max_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = size;
        self
    }

    /// Set the maximum history size.
    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Add a pre-built state.
    pub fn add_state(mut self, state: State<C>) -> Self {
        self.states.push(state);
        self
    }

    /// Add a pre-built transition.
    pub fn add_transition(mut self, transition: Transition<C>) -> Self {
        self.transitions.push(transition);
        self
    }

    /// Build the state machine.
    ///
    /// Returns an error if validation fails.
    pub fn build(self) -> Result<StateMachine<C>, StateMachineError> {
        let mut machine = StateMachine::new(self.name)
            .with_max_queue_size(self.max_queue_size)
            .with_max_history_size(self.max_history_size);

        for state in self.states {
            machine = machine.add_state(state);
        }

        for transition in self.transitions {
            machine = machine.add_transition(transition);
        }

        machine.build()
    }
}

/// Builder for a single state.
pub struct StateBuilder<C> {
    parent: StateMachineBuilder<C>,
    id: StateId,
    is_initial: bool,
    is_final: bool,
    on_entry: Option<ActionFn<C>>,
    on_exit: Option<ActionFn<C>>,
    on_tick: Option<ActionFn<C>>,
    parent_state: Option<StateId>,
    children: Vec<StateId>,
    timeout: Option<Duration>,
    timeout_target: Option<StateId>,
    description: Option<String>,
    metadata: std::collections::HashMap<String, String>,
}

impl<C> StateBuilder<C> {
    fn new(parent: StateMachineBuilder<C>, id: StateId, is_initial: bool) -> Self {
        Self {
            parent,
            id,
            is_initial,
            is_final: false,
            on_entry: None,
            on_exit: None,
            on_tick: None,
            parent_state: None,
            children: Vec::new(),
            timeout: None,
            timeout_target: None,
            description: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Mark this as a final state.
    pub fn final_state(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Set the entry action.
    pub fn on_entry<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.on_entry = Some(Arc::new(action));
        self
    }

    /// Set the exit action.
    pub fn on_exit<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.on_exit = Some(Arc::new(action));
        self
    }

    /// Set the tick action.
    pub fn on_tick<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.on_tick = Some(Arc::new(action));
        self
    }

    /// Set the parent state (for hierarchical FSM).
    pub fn with_parent(mut self, parent: impl Into<StateId>) -> Self {
        self.parent_state = Some(parent.into());
        self
    }

    /// Add a child state.
    pub fn with_child(mut self, child: impl Into<StateId>) -> Self {
        self.children.push(child.into());
        self
    }

    /// Set a timeout for this state.
    pub fn with_timeout(mut self, timeout: Duration, target: impl Into<StateId>) -> Self {
        self.timeout = Some(timeout);
        self.timeout_target = Some(target.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Complete this state and return to the parent builder.
    pub fn done(mut self) -> StateMachineBuilder<C> {
        let mut config = StateConfig::new(self.id.clone());
        config.is_initial = self.is_initial;
        config.is_final = self.is_final;
        config.parent = self.parent_state.clone();
        config.timeout = self.timeout;
        config.timeout_target = self.timeout_target.clone();
        config.description = self.description.clone();
        config.metadata = std::mem::take(&mut self.metadata);

        let state = State {
            config,
            on_entry: self.on_entry,
            on_exit: self.on_exit,
            on_tick: self.on_tick,
            children: self.children,
        };

        self.parent.states.push(state);
        self.parent
    }
}

/// Builder for a single transition.
pub struct TransitionBuilder<C> {
    parent: StateMachineBuilder<C>,
    from: StateId,
    to: StateId,
    event: Option<String>,
    priority: i32,
    guard: Option<GuardFn<C>>,
    actions: Vec<ActionFn<C>>,
    description: Option<String>,
}

impl<C> TransitionBuilder<C> {
    fn new(parent: StateMachineBuilder<C>, from: StateId, to: StateId) -> Self {
        Self {
            parent,
            from,
            to,
            event: None,
            priority: 0,
            guard: None,
            actions: Vec::new(),
            description: None,
        }
    }

    /// Set the triggering event.
    pub fn on_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Add a guard condition.
    pub fn with_guard<F>(mut self, guard: F) -> Self
    where
        F: Fn(&C) -> bool + Send + Sync + 'static,
    {
        self.guard = Some(Arc::new(guard));
        self
    }

    /// Add an action.
    pub fn with_action<F>(mut self, action: F) -> Self
    where
        F: Fn(&mut C) + Send + Sync + 'static,
    {
        self.actions.push(Arc::new(action));
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Complete this transition and return to the parent builder.
    pub fn done(mut self) -> StateMachineBuilder<C> {
        let mut config = TransitionConfig::new(self.from, self.to);
        config.event = self.event;
        config.priority = self.priority;
        config.description = self.description;

        let transition = Transition {
            config,
            guard: self.guard,
            actions: std::mem::take(&mut self.actions),
        };

        self.parent.transitions.push(transition);
        self.parent
    }
}

/// DSL macros for defining state machines more concisely.
#[macro_export]
macro_rules! state_machine {
    (
        $name:ident<$ctx:ty> {
            $(
                state $state_name:ident $(($($state_attr:ident),*))? {
                    $(on_entry: $entry:expr,)?
                    $(on_exit: $exit:expr,)?
                    $(on_tick: $tick:expr,)?
                    $(timeout: $timeout:expr => $timeout_target:ident,)?
                }
            )*

            transitions {
                $(
                    $from:ident -> $to:ident
                        $(on $event:literal)?
                        $(when $guard:expr)?
                        $(=> $action:expr)?
                        ;
                )*
            }
        }
    ) => {{
        let mut builder = $crate::state_machines::StateMachineBuilder::<$ctx>::new(stringify!($name));

        $(
            {
                let state_builder = builder.state(stringify!($state_name));

                // Apply state attributes
                $($(
                    let state_builder = match stringify!($state_attr) {
                        "initial" => state_builder.initial_state(),
                        "final" => state_builder.final_state(),
                        _ => state_builder,
                    };
                )*)?

                // Apply callbacks
                $(let state_builder = state_builder.on_entry($entry);)?
                $(let state_builder = state_builder.on_exit($exit);)?
                $(let state_builder = state_builder.on_tick($tick);)?
                $(let state_builder = state_builder.with_timeout($timeout, stringify!($timeout_target));)?

                builder = state_builder.done();
            }
        )*

        $(
            {
                let mut trans_builder = builder.transition(stringify!($from), stringify!($to));
                $(trans_builder = trans_builder.on_event($event);)?
                $(trans_builder = trans_builder.with_guard($guard);)?
                $(trans_builder = trans_builder.with_action($action);)?
                builder = trans_builder.done();
            }
        )*

        builder.build()
    }};
}

/// Create a simple two-state toggle machine.
pub fn toggle_machine<C: 'static>(
    name: &str,
    off_state: &str,
    on_state: &str,
    toggle_event: &str,
) -> Result<StateMachine<C>, StateMachineError> {
    StateMachineBuilder::<C>::new(name)
        .initial_state(off_state)
        .done()
        .state(on_state)
        .done()
        .transition(off_state, on_state)
        .on_event(toggle_event)
        .done()
        .transition(on_state, off_state)
        .on_event(toggle_event)
        .done()
        .build()
}

/// Create a sequential state machine (linear progression through states).
pub fn sequential_machine<C: 'static>(
    name: &str,
    states: &[&str],
    advance_event: &str,
) -> Result<StateMachine<C>, StateMachineError> {
    if states.is_empty() {
        return Err(StateMachineError::NoInitialState);
    }

    let mut builder = StateMachineBuilder::<C>::new(name);

    // Add first state as initial
    builder = builder.initial_state(states[0]).done();

    // Add remaining states
    for state in &states[1..] {
        builder = builder.state(*state).done();
    }

    // Mark last state as final
    // Note: We'd need to modify the builder for this

    // Add transitions between consecutive states
    for i in 0..states.len() - 1 {
        builder = builder
            .transition(states[i], states[i + 1])
            .on_event(advance_event)
            .done();
    }

    builder.build()
}

/// Create a mode selection state machine with a central "selecting" state.
pub fn mode_selector_machine<C: 'static>(
    name: &str,
    modes: &[&str],
    select_event_prefix: &str,
    return_event: &str,
) -> Result<StateMachine<C>, StateMachineError> {
    if modes.is_empty() {
        return Err(StateMachineError::NoInitialState);
    }

    let mut builder = StateMachineBuilder::<C>::new(name)
        .initial_state("selecting")
        .done();

    // Add mode states and transitions
    for mode in modes {
        builder = builder.state(*mode).done();

        // Transition from selecting to mode
        let event = format!("{}_{}", select_event_prefix, mode);
        builder = builder
            .transition("selecting", *mode)
            .on_event(&event)
            .done();

        // Transition back to selecting
        builder = builder
            .transition(*mode, "selecting")
            .on_event(return_event)
            .done();
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestContext {
        value: i32,
    }

    #[test]
    fn test_builder_basic() {
        let fsm = StateMachineBuilder::<TestContext>::new("test")
            .initial_state("start")
            .done()
            .state("end")
            .final_state()
            .done()
            .transition("start", "end")
            .on_event("go")
            .done()
            .build()
            .unwrap();

        assert_eq!(fsm.name(), "test");
        assert_eq!(fsm.state_ids().len(), 2);
    }

    #[test]
    fn test_builder_with_callbacks() {
        let fsm = StateMachineBuilder::<TestContext>::new("test")
            .initial_state("idle")
            .on_entry(|ctx| ctx.value = 0)
            .on_exit(|ctx| ctx.value = -1)
            .on_tick(|ctx| ctx.value += 1)
            .done()
            .state("running")
            .on_entry(|ctx| ctx.value = 100)
            .done()
            .transition("idle", "running")
            .on_event("start")
            .with_action(|ctx| ctx.value += 50)
            .done()
            .build()
            .unwrap();

        let mut ctx = TestContext { value: 0 };
        let mut fsm = fsm;
        fsm.start(&mut ctx).unwrap();
        assert_eq!(ctx.value, 0); // entry action

        fsm.tick(&mut ctx).unwrap();
        assert_eq!(ctx.value, 1); // tick action

        fsm.process_event(&Event::new("start"), &mut ctx).unwrap();
        // exit (-1) + 1 = 0, then action (+50), then entry (=100)
        assert_eq!(ctx.value, 100);
    }

    #[test]
    fn test_builder_with_guard() {
        let fsm = StateMachineBuilder::<TestContext>::new("test")
            .initial_state("idle")
            .done()
            .state("running")
            .done()
            .transition("idle", "running")
            .on_event("start")
            .with_guard(|ctx| ctx.value > 10)
            .done()
            .build()
            .unwrap();

        let mut ctx = TestContext { value: 5 };
        let mut fsm = fsm;
        fsm.start(&mut ctx).unwrap();

        // Guard should block
        let result = fsm.process_event(&Event::new("start"), &mut ctx).unwrap();
        assert!(!result.is_success());

        // Now it should pass
        ctx.value = 20;
        let result = fsm.process_event(&Event::new("start"), &mut ctx).unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_toggle_machine() {
        let mut fsm = toggle_machine::<TestContext>("toggle", "off", "on", "toggle").unwrap();
        let mut ctx = TestContext { value: 0 };

        fsm.start(&mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("off")));

        fsm.process_event(&Event::new("toggle"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("on")));

        fsm.process_event(&Event::new("toggle"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("off")));
    }

    #[test]
    fn test_sequential_machine() {
        let mut fsm =
            sequential_machine::<TestContext>("seq", &["step1", "step2", "step3"], "next").unwrap();
        let mut ctx = TestContext { value: 0 };

        fsm.start(&mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("step1")));

        fsm.process_event(&Event::new("next"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("step2")));

        fsm.process_event(&Event::new("next"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("step3")));
    }

    #[test]
    fn test_mode_selector_machine() {
        let mut fsm = mode_selector_machine::<TestContext>(
            "modes",
            &["manual", "auto", "calibration"],
            "select",
            "back",
        )
        .unwrap();
        let mut ctx = TestContext { value: 0 };

        fsm.start(&mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("selecting")));

        fsm.process_event(&Event::new("select_auto"), &mut ctx)
            .unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("auto")));

        fsm.process_event(&Event::new("back"), &mut ctx).unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("selecting")));

        fsm.process_event(&Event::new("select_manual"), &mut ctx)
            .unwrap();
        assert_eq!(fsm.current_state(), Some(&StateId::named("manual")));
    }
}
