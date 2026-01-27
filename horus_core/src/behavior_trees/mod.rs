//! HORUS Behavior Trees Module
//!
//! Provides a complete behavior tree implementation for reactive task orchestration
//! in robotics applications. Behavior trees offer a modular, composable approach
//! to robot behavior that is more flexible than finite state machines for
//! complex reactive behaviors.
//!
//! # Key Concepts
//!
//! ## Node Types
//!
//! Behavior trees are composed of different node types:
//!
//! - **Action Nodes**: Leaf nodes that perform actual work (move, sense, compute)
//! - **Condition Nodes**: Leaf nodes that check predicates (battery OK?, obstacle detected?)
//! - **Sequence Nodes**: Execute children in order, fail on first failure
//! - **Selector Nodes**: Execute children in order, succeed on first success
//! - **Parallel Nodes**: Execute all children simultaneously
//! - **Decorator Nodes**: Modify child behavior (invert, repeat, timeout)
//!
//! ## Node Status
//!
//! Every node returns one of three statuses:
//!
//! - `Success`: The node completed successfully
//! - `Failure`: The node failed
//! - `Running`: The node is still executing (will be ticked again)
//!
//! ## Blackboard
//!
//! The blackboard is a shared memory space for nodes to communicate:
//!
//! ```rust,ignore
//! // In one node
//! ctx.blackboard.set("target_position", target);
//!
//! // In another node
//! if let Some(pos) = ctx.blackboard.get::<Position>("target_position") {
//!     // Use the position
//! }
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use horus_core::behavior_trees::{
//!     BehaviorTree, ActionNode, ConditionNode, SequenceNode, SelectorNode, NodeStatus
//! };
//!
//! struct RobotContext {
//!     battery: f32,
//!     target: Option<(f32, f32)>,
//! }
//!
//! // Build a patrol behavior
//! let patrol = SequenceNode::new("patrol")
//!     .add_child(ConditionNode::new("battery_ok", |ctx| {
//!         ctx.context.battery > 0.2
//!     }))
//!     .add_child(ActionNode::new("find_target", |ctx| {
//!         ctx.context.target = Some((10.0, 20.0));
//!         NodeStatus::Success
//!     }))
//!     .add_child(ActionNode::new("move_to_target", |ctx| {
//!         if let Some(_target) = ctx.context.target {
//!             // Move towards target...
//!             NodeStatus::Running
//!         } else {
//!             NodeStatus::Failure
//!         }
//!     }));
//!
//! let mut tree = BehaviorTree::new("robot_patrol", patrol);
//! let mut ctx = RobotContext { battery: 1.0, target: None };
//!
//! // Tick the tree each frame
//! loop {
//!     let status = tree.tick(&mut ctx)?;
//!     if status != NodeStatus::Running {
//!         break;
//!     }
//! }
//! ```
//!
//! # Composite Nodes
//!
//! ## Sequence
//!
//! Executes children left-to-right. Fails immediately if any child fails.
//! Succeeds only if all children succeed.
//!
//! ```rust,ignore
//! let sequence = SequenceNode::new("do_task")
//!     .add_child(check_preconditions)
//!     .add_child(execute_action)
//!     .add_child(verify_result);
//! ```
//!
//! ## Selector (Fallback)
//!
//! Executes children left-to-right. Succeeds immediately if any child succeeds.
//! Fails only if all children fail.
//!
//! ```rust,ignore
//! let selector = SelectorNode::new("find_path")
//!     .add_child(use_cached_path)
//!     .add_child(compute_new_path)
//!     .add_child(request_human_help);
//! ```
//!
//! ## Parallel
//!
//! Executes all children simultaneously. Uses a policy to determine success/failure.
//!
//! ```rust,ignore
//! let parallel = ParallelNode::new("monitor_and_move", ParallelPolicy::RequireAll)
//!     .add_child(monitor_obstacles)
//!     .add_child(execute_trajectory);
//! ```
//!
//! # Decorator Nodes
//!
//! Decorators wrap a single child and modify its behavior:
//!
//! ```rust,ignore
//! // Invert the result
//! let not_blocked = DecoratorNode::inverter("not_blocked", check_blocked);
//!
//! // Retry up to 3 times
//! let retry = DecoratorNode::repeater("retry", 3, attempt_connection);
//!
//! // Timeout after 5 seconds
//! let timed = DecoratorNode::timeout("timed", Duration::from_secs(5), long_action);
//!
//! // Delay before executing
//! let delayed = DecoratorNode::delay("delayed", Duration::from_secs(1), action);
//!
//! // Cooldown between executions
//! let rate_limited = DecoratorNode::cooldown("rate_limited", Duration::from_secs(5), action);
//! ```
//!
//! # Reactive Variants
//!
//! Reactive sequences and selectors re-evaluate from the beginning each tick,
//! allowing higher-priority conditions to preempt running behaviors:
//!
//! ```rust,ignore
//! let reactive = ReactiveSequenceNode::new("safe_move")
//!     .add_child(ConditionNode::new("no_obstacle", |ctx| !ctx.context.obstacle_detected))
//!     .add_child(ActionNode::new("move", |ctx| {
//!         // This will be interrupted if obstacle appears
//!         NodeStatus::Running
//!     }));
//! ```
//!
//! # Tree Builder
//!
//! Use the builder for fluent tree construction:
//!
//! ```rust,ignore
//! let tree = BehaviorTreeBuilder::<RobotContext>::new("robot_ai")
//!     .with_tick_rate(Duration::from_millis(100))
//!     .reset_on_complete(true)
//!     .root(
//!         SelectorNode::new("main")
//!             .add_child(emergency_behavior)
//!             .add_child(normal_operation)
//!     )
//!     .build()?;
//! ```
//!
//! # Metrics and Monitoring
//!
//! Trees track execution metrics:
//!
//! ```rust,ignore
//! let metrics = tree.metrics();
//! println!("Total ticks: {}", metrics.total_ticks);
//! println!("Success rate: {:.1}%",
//!     metrics.success_count as f64 / metrics.total_ticks as f64 * 100.0);
//! ```
//!
//! # Visualization
//!
//! Generate ASCII tree visualization for debugging:
//!
//! ```rust,ignore
//! let ascii = TreeVisualizer::to_ascii(&tree);
//! println!("{}", ascii);
//! ```

pub mod nodes;
pub mod tree;
pub mod types;

// Re-export main types at module level
pub use nodes::{
    ActionFn, ActionNode, BTNode, ConditionFn, ConditionNode, DecoratorNode, GuardFn, ParallelNode,
    ReactiveSelectorNode, ReactiveSequenceNode, SelectorNode, SequenceNode, SubtreeNode,
};
pub use tree::{BehaviorTree, BehaviorTreeBuilder, NodeStats, SharedBehaviorTree, TreeVisualizer};
pub use types::{
    BehaviorTreeConfig, BehaviorTreeError, BehaviorTreeMetrics, Blackboard, BlackboardValue,
    DecoratorType, NodeId, NodeStatus, NodeType, ParallelPolicy, TickContext,
};

/// Prelude module for convenient imports.
///
/// ```rust,ignore
/// use horus_core::behavior_trees::prelude::*;
/// ```
pub mod prelude {
    pub use super::{
        ActionNode, BTNode, BehaviorTree, BehaviorTreeBuilder, Blackboard, ConditionNode,
        DecoratorNode, DecoratorType, NodeId, NodeStatus, NodeType, ParallelNode, ParallelPolicy,
        ReactiveSelectorNode, ReactiveSequenceNode, SelectorNode, SequenceNode, SharedBehaviorTree,
        TickContext, TreeVisualizer,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct TestContext {
        value: i32,
        flag: bool,
    }

    #[test]
    fn test_module_exports() {
        // Verify main types are accessible
        let _status = NodeStatus::Success;
        let _id = NodeId::new("test");
        let _blackboard = Blackboard::new();
    }

    #[test]
    fn test_simple_tree() {
        let root = SequenceNode::new("main")
            .add_child(ConditionNode::new(
                "check",
                |ctx: &TickContext<TestContext>| ctx.context.flag,
            ))
            .add_child(ActionNode::new(
                "action",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.value = 42;
                    NodeStatus::Success
                },
            ));

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext {
            value: 0,
            flag: true,
        };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.value, 42);
    }

    #[test]
    fn test_selector_fallback() {
        let root = SelectorNode::new("fallback")
            .add_child(ActionNode::new(
                "fail",
                |_: &mut TickContext<TestContext>| NodeStatus::Failure,
            ))
            .add_child(ActionNode::new(
                "succeed",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.value = 100;
                    NodeStatus::Success
                },
            ));

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext {
            value: 0,
            flag: false,
        };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.value, 100);
    }

    #[test]
    fn test_parallel_node() {
        let root = ParallelNode::new("parallel", ParallelPolicy::RequireAll)
            .add_child(ActionNode::new(
                "a1",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.value += 1;
                    NodeStatus::Success
                },
            ))
            .add_child(ActionNode::new(
                "a2",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.value += 10;
                    NodeStatus::Success
                },
            ));

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext {
            value: 0,
            flag: false,
        };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.value, 11);
    }

    #[test]
    fn test_decorator_inverter() {
        let root = DecoratorNode::inverter(
            "invert",
            ActionNode::new("fail", |_: &mut TickContext<TestContext>| {
                NodeStatus::Failure
            }),
        );

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext {
            value: 0,
            flag: false,
        };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
    }

    #[test]
    fn test_blackboard_communication() {
        let root = SequenceNode::new("main")
            .add_child(ActionNode::new(
                "write",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.blackboard.set("key", 42i32);
                    NodeStatus::Success
                },
            ))
            .add_child(ActionNode::new(
                "read",
                |ctx: &mut TickContext<TestContext>| {
                    if let Some(value) = ctx.blackboard.get_int("key") {
                        ctx.context.value = value as i32;
                        NodeStatus::Success
                    } else {
                        NodeStatus::Failure
                    }
                },
            ));

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext {
            value: 0,
            flag: false,
        };

        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Success);
        assert_eq!(ctx.value, 42);
    }

    #[test]
    fn test_builder_pattern() {
        let tree = BehaviorTreeBuilder::<TestContext>::new("builder_test")
            .with_tick_rate(Duration::from_millis(50))
            .reset_on_complete(true)
            .collect_metrics(true)
            .root(ActionNode::new(
                "test",
                |_: &mut TickContext<TestContext>| NodeStatus::Success,
            ))
            .build()
            .unwrap();

        assert_eq!(tree.name(), "builder_test");
        assert!(tree.config().reset_on_complete);
        assert!(tree.config().collect_metrics);
    }

    #[test]
    fn test_shared_tree() {
        let root = ActionNode::new("inc", |ctx: &mut TickContext<TestContext>| {
            ctx.context.value += 1;
            NodeStatus::Success
        });

        let tree = BehaviorTree::new("shared", root);
        let shared = SharedBehaviorTree::new(tree);

        let mut ctx = TestContext {
            value: 0,
            flag: false,
        };

        // Tick from multiple references
        shared.tick(&mut ctx).unwrap();
        let clone = shared.clone_arc();
        clone.tick(&mut ctx).unwrap();

        assert_eq!(ctx.value, 2);
    }

    #[test]
    fn test_reactive_sequence() {
        // A reactive sequence that checks a condition every tick
        let root = ReactiveSequenceNode::new("reactive")
            .add_child(ConditionNode::new(
                "flag_set",
                |ctx: &TickContext<TestContext>| ctx.context.flag,
            ))
            .add_child(ActionNode::new(
                "inc",
                |ctx: &mut TickContext<TestContext>| {
                    ctx.context.value += 1;
                    NodeStatus::Running
                },
            ));

        let mut tree = BehaviorTree::new("test", root);
        let mut ctx = TestContext {
            value: 0,
            flag: true,
        };

        // First tick: condition passes, action runs
        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Running);
        assert_eq!(ctx.value, 1);

        // Second tick with flag still true
        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Running);
        assert_eq!(ctx.value, 2);

        // Third tick with flag false - should fail at condition
        ctx.flag = false;
        let status = tree.tick(&mut ctx).unwrap();
        assert_eq!(status, NodeStatus::Failure);
        assert_eq!(ctx.value, 2); // Value unchanged
    }

    #[test]
    fn test_tree_visualization() {
        let root = SequenceNode::<TestContext>::new("root")
            .add_child(ActionNode::new(
                "child1",
                |_: &mut TickContext<TestContext>| NodeStatus::Success,
            ))
            .add_child(
                SelectorNode::new("child2")
                    .add_child(ActionNode::new(
                        "grandchild1",
                        |_: &mut TickContext<TestContext>| NodeStatus::Failure,
                    ))
                    .add_child(ActionNode::new(
                        "grandchild2",
                        |_: &mut TickContext<TestContext>| NodeStatus::Success,
                    )),
            );

        let tree = BehaviorTree::new("viz_test", root);
        let ascii = TreeVisualizer::to_ascii(&tree);

        // Verify structure is present
        assert!(ascii.contains("viz_test"));
        assert!(ascii.contains("root"));
        assert!(ascii.contains("child1"));
        assert!(ascii.contains("child2"));
        assert!(ascii.contains("grandchild1"));
        assert!(ascii.contains("grandchild2"));
    }
}
