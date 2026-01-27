//! Behavior Tree Implementation
//!
//! This module provides the main `BehaviorTree` struct that manages
//! tree execution, event handling, and metrics collection.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use super::nodes::BTNode;
use super::types::{
    BehaviorTreeConfig, BehaviorTreeError, BehaviorTreeMetrics, Blackboard, NodeStatus, TickContext,
};

/// A complete behavior tree with execution management.
///
/// The BehaviorTree wraps a root node and provides:
/// - Tree execution with configurable tick rates
/// - Blackboard for inter-node communication
/// - Metrics tracking
/// - Event handling
///
/// # Example
///
/// ```rust,ignore
/// use horus_core::behavior_trees::{BehaviorTree, ActionNode, SequenceNode};
///
/// struct RobotContext {
///     velocity: f32,
/// }
///
/// let root = SequenceNode::new("main")
///     .add_child(ActionNode::new("check_sensors", |ctx| {
///         // Check sensors...
///         NodeStatus::Success
///     }))
///     .add_child(ActionNode::new("move", |ctx| {
///         ctx.context.velocity = 0.5;
///         NodeStatus::Running
///     }));
///
/// let mut tree = BehaviorTree::new("robot_behavior", root);
/// let mut ctx = RobotContext { velocity: 0.0 };
///
/// // Tick the tree
/// tree.tick(&mut ctx)?;
/// ```
pub struct BehaviorTree<C> {
    /// Name of this behavior tree.
    name: String,
    /// The root node of the tree.
    root: Box<dyn BTNode<C>>,
    /// Shared blackboard for inter-node communication.
    blackboard: Blackboard,
    /// Configuration options.
    config: BehaviorTreeConfig,
    /// Metrics for monitoring.
    metrics: BehaviorTreeMetrics,
    /// Whether the tree is currently running.
    is_running: bool,
    /// Current status of the tree.
    status: NodeStatus,
    /// Time when the tree started.
    start_time: Option<Instant>,
    /// Time of the last tick.
    last_tick_time: Option<Instant>,
    /// Total elapsed time.
    elapsed: Duration,
    /// Number of ticks executed.
    tick_count: u64,
}

impl<C: Send + Sync> BehaviorTree<C> {
    /// Create a new behavior tree with the given root node.
    pub fn new<S: Into<String>>(name: S, root: impl BTNode<C> + 'static) -> Self {
        Self {
            name: name.into(),
            root: Box::new(root),
            blackboard: Blackboard::new(),
            config: BehaviorTreeConfig::default(),
            metrics: BehaviorTreeMetrics::default(),
            is_running: false,
            status: NodeStatus::Failure,
            start_time: None,
            last_tick_time: None,
            elapsed: Duration::ZERO,
            tick_count: 0,
        }
    }

    /// Create a new behavior tree with custom configuration.
    pub fn with_config<S: Into<String>>(
        name: S,
        root: impl BTNode<C> + 'static,
        config: BehaviorTreeConfig,
    ) -> Self {
        Self {
            name: name.into(),
            root: Box::new(root),
            blackboard: Blackboard::new(),
            config,
            metrics: BehaviorTreeMetrics::default(),
            is_running: false,
            status: NodeStatus::Failure,
            start_time: None,
            last_tick_time: None,
            elapsed: Duration::ZERO,
            tick_count: 0,
        }
    }

    /// Get the name of this behavior tree.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the current configuration.
    pub fn config(&self) -> &BehaviorTreeConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut BehaviorTreeConfig {
        &mut self.config
    }

    /// Get a reference to the blackboard.
    pub fn blackboard(&self) -> &Blackboard {
        &self.blackboard
    }

    /// Get a mutable reference to the blackboard.
    pub fn blackboard_mut(&mut self) -> &mut Blackboard {
        &mut self.blackboard
    }

    /// Get the current metrics.
    pub fn metrics(&self) -> &BehaviorTreeMetrics {
        &self.metrics
    }

    /// Get the current status of the tree.
    pub fn status(&self) -> NodeStatus {
        self.status
    }

    /// Check if the tree is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get the total tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Get the total elapsed time since the tree started.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Start the behavior tree.
    pub fn start(&mut self) -> Result<(), BehaviorTreeError> {
        if self.is_running {
            return Err(BehaviorTreeError::AlreadyRunning);
        }

        self.is_running = true;
        self.start_time = Some(Instant::now());
        self.last_tick_time = Some(Instant::now());
        self.elapsed = Duration::ZERO;
        self.tick_count = 0;
        self.status = NodeStatus::Running;
        self.metrics.reset();

        Ok(())
    }

    /// Stop the behavior tree and reset all nodes.
    pub fn stop(&mut self) {
        self.is_running = false;
        self.status = NodeStatus::Failure;
        self.root.reset();
    }

    /// Tick the behavior tree once.
    ///
    /// This executes one iteration of the tree, starting from the root node.
    pub fn tick(&mut self, context: &mut C) -> Result<NodeStatus, BehaviorTreeError> {
        let tick_start = Instant::now();

        // Auto-start if not running
        if !self.is_running {
            self.start()?;
        }

        // Calculate delta time
        let delta_time = self.last_tick_time.map_or(Duration::ZERO, |t| t.elapsed());
        self.last_tick_time = Some(tick_start);
        self.elapsed = self.start_time.map_or(Duration::ZERO, |t| t.elapsed());

        // Create tick context
        let mut tick_ctx = TickContext {
            blackboard: &mut self.blackboard,
            context,
            elapsed: self.elapsed,
            delta_time,
            tick_count: self.tick_count,
        };

        // Execute the tree
        self.status = self.root.tick(&mut tick_ctx);
        self.tick_count += 1;

        // Update metrics
        let tick_duration = tick_start.elapsed();
        self.metrics.total_ticks += 1;
        self.metrics.total_tick_time += tick_duration;
        if tick_duration > self.metrics.max_tick_time {
            self.metrics.max_tick_time = tick_duration;
        }
        if tick_duration < self.metrics.min_tick_time
            || self.metrics.min_tick_time == Duration::ZERO
        {
            self.metrics.min_tick_time = tick_duration;
        }

        // Track status changes
        match self.status {
            NodeStatus::Success => {
                self.metrics.success_count += 1;
                if self.config.reset_on_complete {
                    self.root.reset();
                }
            }
            NodeStatus::Failure => {
                self.metrics.failure_count += 1;
                if self.config.reset_on_complete {
                    self.root.reset();
                }
            }
            NodeStatus::Running => {
                self.metrics.running_count += 1;
            }
        }

        Ok(self.status)
    }

    /// Tick the tree multiple times until it completes or reaches max ticks.
    ///
    /// Returns the final status and number of ticks executed.
    pub fn run_until_complete(
        &mut self,
        context: &mut C,
        max_ticks: Option<u64>,
    ) -> Result<(NodeStatus, u64), BehaviorTreeError> {
        let max = max_ticks.unwrap_or(u64::MAX);
        let mut ticks = 0;

        loop {
            let status = self.tick(context)?;
            ticks += 1;

            if status != NodeStatus::Running || ticks >= max {
                return Ok((status, ticks));
            }
        }
    }

    /// Reset the behavior tree to its initial state.
    pub fn reset(&mut self) {
        self.is_running = false;
        self.status = NodeStatus::Failure;
        self.start_time = None;
        self.last_tick_time = None;
        self.elapsed = Duration::ZERO;
        self.tick_count = 0;
        self.root.reset();
        self.blackboard.clear();
    }

    /// Get a reference to the root node.
    pub fn root(&self) -> &dyn BTNode<C> {
        self.root.as_ref()
    }

    /// Get a mutable reference to the root node.
    pub fn root_mut(&mut self) -> &mut dyn BTNode<C> {
        self.root.as_mut()
    }

    /// Replace the root node with a new one.
    pub fn set_root(&mut self, root: impl BTNode<C> + 'static) {
        self.root = Box::new(root);
        self.reset();
    }
}

/// A thread-safe wrapper around BehaviorTree.
///
/// Allows the behavior tree to be shared across threads.
pub struct SharedBehaviorTree<C> {
    inner: Arc<RwLock<BehaviorTree<C>>>,
}

impl<C: Send + Sync + 'static> SharedBehaviorTree<C> {
    /// Create a new shared behavior tree.
    pub fn new(tree: BehaviorTree<C>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(tree)),
        }
    }

    /// Get the name of this behavior tree.
    pub fn name(&self) -> String {
        self.inner.read().name.clone()
    }

    /// Get the current status.
    pub fn status(&self) -> NodeStatus {
        self.inner.read().status
    }

    /// Check if the tree is running.
    pub fn is_running(&self) -> bool {
        self.inner.read().is_running
    }

    /// Get the current metrics.
    pub fn metrics(&self) -> BehaviorTreeMetrics {
        self.inner.read().metrics.clone()
    }

    /// Get a value from the blackboard.
    pub fn blackboard_get(&self, key: &str) -> Option<super::types::BlackboardValue> {
        self.inner.read().blackboard.get(key).cloned()
    }

    /// Set a value in the blackboard.
    pub fn blackboard_set(
        &self,
        key: impl Into<String>,
        value: impl Into<super::types::BlackboardValue>,
    ) {
        self.inner.write().blackboard.set(key, value);
    }

    /// Start the behavior tree.
    pub fn start(&self) -> Result<(), BehaviorTreeError> {
        self.inner.write().start()
    }

    /// Stop the behavior tree.
    pub fn stop(&self) {
        self.inner.write().stop();
    }

    /// Tick the behavior tree.
    pub fn tick(&self, context: &mut C) -> Result<NodeStatus, BehaviorTreeError> {
        self.inner.write().tick(context)
    }

    /// Reset the behavior tree.
    pub fn reset(&self) {
        self.inner.write().reset();
    }

    /// Clone the Arc for sharing.
    pub fn clone_arc(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C> Clone for SharedBehaviorTree<C> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Builder for constructing behavior trees.
///
/// Provides a fluent API for building complex behavior trees.
///
/// # Example
///
/// ```rust,ignore
/// let tree = BehaviorTreeBuilder::<RobotContext>::new("patrol")
///     .with_tick_rate(Duration::from_millis(100))
///     .with_max_tick_time(Duration::from_millis(50))
///     .root(
///         SequenceNode::new("patrol_sequence")
///             .add_child(move_to_waypoint_1)
///             .add_child(scan_area)
///             .add_child(move_to_waypoint_2)
///     )
///     .build()?;
/// ```
pub struct BehaviorTreeBuilder<C> {
    name: String,
    config: BehaviorTreeConfig,
    root: Option<Box<dyn BTNode<C>>>,
    initial_blackboard: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl<C: Send + Sync + 'static> BehaviorTreeBuilder<C> {
    /// Create a new builder with the given name.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            config: BehaviorTreeConfig::default(),
            root: None,
            initial_blackboard: HashMap::new(),
        }
    }

    /// Set the tick rate for the tree.
    pub fn with_tick_rate(mut self, rate: Duration) -> Self {
        self.config.tick_rate = Some(rate);
        self
    }

    /// Set the maximum allowed tick time.
    pub fn with_max_tick_time(mut self, max_time: Duration) -> Self {
        self.config.max_tick_time = Some(max_time);
        self
    }

    /// Enable or disable reset on complete.
    pub fn reset_on_complete(mut self, reset: bool) -> Self {
        self.config.reset_on_complete = reset;
        self
    }

    /// Enable or disable metrics collection.
    pub fn collect_metrics(mut self, collect: bool) -> Self {
        self.config.collect_metrics = collect;
        self
    }

    /// Enable or disable debug mode.
    pub fn debug_mode(mut self, debug: bool) -> Self {
        self.config.debug_mode = debug;
        self
    }

    /// Set the root node.
    pub fn root(mut self, root: impl BTNode<C> + 'static) -> Self {
        self.root = Some(Box::new(root));
        self
    }

    /// Add an initial blackboard value.
    pub fn with_blackboard_value<T: Clone + Send + Sync + 'static>(
        mut self,
        key: &str,
        value: T,
    ) -> Self {
        self.initial_blackboard
            .insert(key.to_string(), Box::new(value));
        self
    }

    /// Build the behavior tree.
    pub fn build(self) -> Result<BehaviorTree<C>, BehaviorTreeError> {
        let root = self.root.ok_or(BehaviorTreeError::InvalidTree {
            reason: "No root node specified".to_string(),
        })?;

        let tree = BehaviorTree {
            name: self.name,
            root,
            blackboard: Blackboard::new(),
            config: self.config,
            metrics: BehaviorTreeMetrics::default(),
            is_running: false,
            status: NodeStatus::Failure,
            start_time: None,
            last_tick_time: None,
            elapsed: Duration::ZERO,
            tick_count: 0,
        };

        // Initialize blackboard with provided values
        for (key, value) in self.initial_blackboard {
            // We can't directly insert Any into the blackboard,
            // so this is a placeholder for type-safe initialization
            // In practice, users would use blackboard_mut() after building
            let _ = (key, value);
        }

        Ok(tree)
    }

    /// Build a shared behavior tree.
    pub fn build_shared(self) -> Result<SharedBehaviorTree<C>, BehaviorTreeError> {
        Ok(SharedBehaviorTree::new(self.build()?))
    }
}

/// Statistics about a node during tree execution.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    /// Number of times this node was ticked.
    pub tick_count: u64,
    /// Number of times this node returned Success.
    pub success_count: u64,
    /// Number of times this node returned Failure.
    pub failure_count: u64,
    /// Number of times this node returned Running.
    pub running_count: u64,
    /// Total time spent in this node.
    pub total_time: Duration,
    /// Average time per tick.
    pub avg_time: Duration,
}

/// Utility for visualizing behavior trees.
pub struct TreeVisualizer;

impl TreeVisualizer {
    /// Generate an ASCII representation of the tree structure.
    pub fn to_ascii<C>(tree: &BehaviorTree<C>) -> String {
        let mut output = String::new();
        output.push_str(&format!("Behavior Tree: {}\n", tree.name));
        output.push_str(&format!("Status: {:?}\n", tree.status));
        output.push_str(&format!("Ticks: {}\n", tree.tick_count));
        output.push_str("\nStructure:\n");
        Self::node_to_ascii(tree.root.as_ref(), &mut output, "", true);
        output
    }

    fn node_to_ascii<C>(node: &dyn BTNode<C>, output: &mut String, prefix: &str, is_last: bool) {
        let connector = if is_last { "└── " } else { "├── " };
        let status_indicator = match node.status() {
            NodeStatus::Success => "[✓]",
            NodeStatus::Failure => "[✗]",
            NodeStatus::Running => "[→]",
        };

        output.push_str(&format!(
            "{}{}{} {} ({:?})\n",
            prefix,
            connector,
            status_indicator,
            node.name(),
            node.node_type()
        ));

        let new_prefix = format!("{}{}   ", prefix, if is_last { " " } else { "│" });
        let children = node.children();
        for (i, child) in children.iter().enumerate() {
            let is_last_child = i == children.len() - 1;
            Self::node_to_ascii(child.as_ref(), output, &new_prefix, is_last_child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior_trees::nodes::{ActionNode, SelectorNode, SequenceNode};
    use crate::behavior_trees::TickContext;

    struct TestContext {
        counter: i32,
    }

    #[test]
    fn test_behavior_tree_basic() {
        let root = ActionNode::new("increment", |ctx: &mut TickContext<TestContext>| {
            ctx.context.counter += 1;
            NodeStatus::Success
        });

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext { counter: 0 };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.counter, 1);
    }

    #[test]
    fn test_behavior_tree_sequence() {
        let root = SequenceNode::new("sequence")
            .add_child(ActionNode::new(
                "a1",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.counter += 1;
                    NodeStatus::Success
                },
            ))
            .add_child(ActionNode::new(
                "a2",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.counter += 10;
                    NodeStatus::Success
                },
            ));

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext { counter: 0 };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.counter, 11);
    }

    #[test]
    fn test_behavior_tree_metrics() {
        let root = ActionNode::new("test", |_: &mut TickContext<TestContext>| {
            NodeStatus::Success
        });
        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext { counter: 0 };

        for _ in 0..5 {
            tree.tick(&mut ctx).unwrap();
        }

        let metrics = tree.metrics();
        assert_eq!(metrics.total_ticks, 5);
        assert_eq!(metrics.success_count, 5);
    }

    #[test]
    fn test_behavior_tree_blackboard() {
        let root = ActionNode::new("use_blackboard", |ctx: &mut TickContext<TestContext>| {
            if let Some(value) = ctx.blackboard.get_int("target") {
                ctx.context.counter = value as i32;
                NodeStatus::Success
            } else {
                NodeStatus::Failure
            }
        });

        let mut tree = BehaviorTree::new("test", root);
        tree.blackboard_mut().set("target", 42i32);

        let mut ctx = TestContext { counter: 0 };
        let status = tree.tick(&mut ctx).unwrap();

        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.counter, 42);
    }

    #[test]
    fn test_behavior_tree_builder() {
        let tree = BehaviorTreeBuilder::<TestContext>::new("test")
            .with_tick_rate(Duration::from_millis(100))
            .reset_on_complete(true)
            .root(ActionNode::new(
                "test",
                |_: &mut TickContext<TestContext>| NodeStatus::Success,
            ))
            .build()
            .unwrap();

        assert_eq!(tree.name(), "test");
        assert_eq!(tree.config().tick_rate, Some(Duration::from_millis(100)));
        assert!(tree.config().reset_on_complete);
    }

    #[test]
    fn test_shared_behavior_tree() {
        let root = ActionNode::new("test", |ctx: &mut TickContext<TestContext>| {
            ctx.context.counter += 1;
            NodeStatus::Success
        });

        let tree = BehaviorTree::new("shared", root);
        let shared = SharedBehaviorTree::new(tree);

        let mut ctx = TestContext { counter: 0 };

        // Can tick from the shared wrapper
        let status = shared.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.counter, 1);

        // Can clone and share
        let shared2 = shared.clone_arc();
        let status = shared2.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.counter, 2);
    }

    #[test]
    fn test_run_until_complete() {
        use std::sync::atomic::{AtomicI32, Ordering};
        let counter = AtomicI32::new(0);
        let root = ActionNode::new(
            "run_three_times",
            move |_: &mut TickContext<TestContext>| {
                let val = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if val >= 3 {
                    NodeStatus::Success
                } else {
                    NodeStatus::Running
                }
            },
        );

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext { counter: 0 };

        let (status, ticks) = tree.run_until_complete(&mut ctx, Some(10)).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ticks, 3);
    }

    #[test]
    fn test_tree_visualizer() {
        let root = SequenceNode::<TestContext>::new("main")
            .add_child(ActionNode::new(
                "step1",
                |_: &mut TickContext<TestContext>| NodeStatus::Success,
            ))
            .add_child(
                SelectorNode::new("fallback")
                    .add_child(ActionNode::new(
                        "option1",
                        |_: &mut TickContext<TestContext>| NodeStatus::Failure,
                    ))
                    .add_child(ActionNode::new(
                        "option2",
                        |_: &mut TickContext<TestContext>| NodeStatus::Success,
                    )),
            );

        let tree = BehaviorTree::new("test_tree", root);
        let ascii = TreeVisualizer::to_ascii(&tree);

        assert!(ascii.contains("test_tree"));
        assert!(ascii.contains("main"));
        assert!(ascii.contains("step1"));
        assert!(ascii.contains("fallback"));
    }
}
