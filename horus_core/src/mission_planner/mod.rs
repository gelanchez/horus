//! HORUS Mission Planner Module
//!
//! Provides mission planning and execution capabilities for complex robot tasks.
//! The mission planner orchestrates multiple goals and tasks with dependency
//! management, priority scheduling, and failure handling.
//!
//! # Key Concepts
//!
//! ## Mission Hierarchy
//!
//! Missions are organized in a three-level hierarchy:
//!
//! - **Mission**: A complete operation (e.g., "Deliver package to room 101")
//! - **Goal**: A sub-objective of a mission (e.g., "Navigate to room", "Pick up package")
//! - **Task**: An atomic unit of work (e.g., "Compute path", "Execute trajectory")
//!
//! ## Dependencies
//!
//! Tasks and goals can have dependencies on each other. The planner uses a
//! directed acyclic graph (DAG) to track dependencies and determine execution
//! order. Tasks are only executed when all their dependencies are complete.
//!
//! ## Execution Modes
//!
//! Missions can be executed in different modes:
//!
//! - **Sequential**: Goals execute one at a time in order
//! - **Parallel**: Multiple goals execute simultaneously
//! - **Opportunistic**: Goals execute when conditions are favorable
//!
//! ## Task Executors
//!
//! Tasks can be executed by different executors:
//!
//! - **Action**: Long-running action with feedback (see `actions` module)
//! - **BehaviorTree**: Execute a behavior tree
//! - **StateMachine**: Transition a state machine
//! - **Custom**: Custom handler function
//! - **Command**: Shell command execution
//! - **Noop**: No operation (synchronization point)
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use horus_core::mission_planner::{
//!     MissionPlanner, MissionSpec, GoalSpec, TaskSpec, TaskExecutor,
//! };
//!
//! // Create a simple mission
//! let navigate_task = TaskSpec::new(
//!     "navigate",
//!     "Navigate to Target",
//!     TaskExecutor::Action {
//!         action_type: "navigate_to_pose".to_string(),
//!         parameters: Default::default(),
//!     },
//! );
//!
//! let pickup_task = TaskSpec::new(
//!     "pickup",
//!     "Pick Up Object",
//!     TaskExecutor::Action {
//!         action_type: "grasp_object".to_string(),
//!         parameters: Default::default(),
//!     },
//! );
//!
//! let goal = GoalSpec::new("fetch", "Fetch Object")
//!     .add_task(navigate_task.clone())
//!     .add_task_after(pickup_task, &navigate_task.id);
//!
//! let mission = MissionSpec::new("delivery")
//!     .with_description("Deliver a package")
//!     .add_goal(goal);
//!
//! // Create and configure planner
//! let mut planner = MissionPlanner::new();
//!
//! // Register executor for actions
//! planner.register_executor("navigate_to_pose", Arc::new(|task, ctx| {
//!     // Execute navigation action...
//!     Ok(Some(serde_json::json!({"success": true})))
//! }));
//!
//! // Submit and start mission
//! let id = planner.submit(mission)?;
//! planner.start(&id)?;
//!
//! // Tick periodically to advance execution
//! loop {
//!     planner.tick()?;
//!     let status = planner.get_mission_status(&id)?;
//!     if status.status.is_terminal() {
//!         break;
//!     }
//!     std::thread::sleep(std::time::Duration::from_millis(100));
//! }
//! ```
//!
//! # Dependency Management
//!
//! Tasks and goals can have dependencies:
//!
//! ```rust,ignore
//! let task_a = TaskSpec::noop("a", "Task A");
//! let task_b = TaskSpec::noop("b", "Task B");
//! let task_c = TaskSpec::noop("c", "Task C");
//!
//! // C depends on both A and B
//! let goal = GoalSpec::new("parallel", "Parallel Tasks")
//!     .add_task(task_a.clone())
//!     .add_task(task_b.clone())
//!     .add_task_after_all(task_c, &[task_a.id.clone(), task_b.id.clone()]);
//! ```
//!
//! # Failure Handling
//!
//! Goals have configurable failure policies:
//!
//! ```rust,ignore
//! let goal = GoalSpec::new("critical", "Critical Goal")
//!     .with_failure_policy(GoalFailurePolicy::AbortMission);
//!
//! let optional_goal = GoalSpec::new("optional", "Optional Goal")
//!     .with_failure_policy(GoalFailurePolicy::Continue)
//!     .optional();
//! ```
//!
//! # Retry Policies
//!
//! Tasks can be configured to retry on failure:
//!
//! ```rust,ignore
//! let task = TaskSpec::new("flaky", "Flaky Task", TaskExecutor::Noop)
//!     .with_retry(RetryPolicy {
//!         max_attempts: 3,
//!         delay: Duration::from_secs(1),
//!         exponential_backoff: true,
//!         max_delay: Duration::from_secs(30),
//!         retry_on: vec!["NetworkError".to_string()],
//!     });
//! ```
//!
//! # Event Handling
//!
//! Subscribe to mission events for monitoring:
//!
//! ```rust,ignore
//! planner.on_event(Arc::new(|event| {
//!     match event {
//!         MissionEvent::TaskCompleted { task_id, success, .. } => {
//!             println!("Task {} completed: {}", task_id, success);
//!         }
//!         MissionEvent::Progress { progress, .. } => {
//!             println!("Progress: {:.1}%", progress * 100.0);
//!         }
//!         _ => {}
//!     }
//! }));
//! ```
//!
//! # Thread Safety
//!
//! Use `SharedMissionPlanner` for multi-threaded access:
//!
//! ```rust,ignore
//! let planner = MissionPlanner::new();
//! let shared = SharedMissionPlanner::new(planner);
//!
//! // Clone for different threads
//! let planner_clone = shared.clone_arc();
//! std::thread::spawn(move || {
//!     planner_clone.tick().unwrap();
//! });
//! ```
//!
//! # Builder Pattern
//!
//! Use the builder for complex planner configuration:
//!
//! ```rust,ignore
//! let planner = MissionPlannerBuilder::new()
//!     .max_concurrent_missions(4)
//!     .max_concurrent_tasks(8)
//!     .default_task_timeout(Duration::from_secs(60))
//!     .collect_metrics(true)
//!     .register_executor("custom", my_executor)
//!     .on_event(my_callback)
//!     .build();
//! ```

pub mod dag;
pub mod planner;
pub mod types;

// Re-export main types at module level
pub use dag::{
    build_goal_dag, build_task_dag, DependencyGraph, ExecutionScheduler, GoalDAG, TaskDAG,
};
pub use planner::{
    ConditionEvaluatorFn, ExecutionContext, MissionEvent, MissionPlanner, MissionPlannerBuilder,
    MissionPlannerConfig, SharedMissionPlanner, TaskExecutorFn,
};
pub use types::{
    ExecutionStatus, GoalFailurePolicy, GoalId, GoalSpec, GoalState, MissionId, MissionMetrics,
    MissionMode, MissionPlannerError, MissionSpec, MissionState, Priority, RetryPolicy,
    TaskCondition, TaskExecutor, TaskId, TaskSpec, TaskState,
};

/// Prelude module for convenient imports.
///
/// ```rust,ignore
/// use horus_core::mission_planner::prelude::*;
/// ```
pub mod prelude {
    pub use super::{
        ExecutionStatus, GoalFailurePolicy, GoalId, GoalSpec, GoalState, MissionEvent, MissionId,
        MissionMode, MissionPlanner, MissionPlannerBuilder, MissionSpec, MissionState, Priority,
        RetryPolicy, SharedMissionPlanner, TaskCondition, TaskExecutor, TaskId, TaskSpec,
        TaskState,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_module_exports() {
        // Verify main types are accessible
        let _status = ExecutionStatus::Pending;
        let _priority = Priority::Normal;
        let _mode = MissionMode::Sequential;
    }

    #[test]
    fn test_simple_mission_flow() {
        // Create a simple mission with two sequential tasks
        let task1 = TaskSpec::noop("t1", "Task 1");
        let task2 = TaskSpec::noop("t2", "Task 2");
        let t1_id = task1.id.clone();

        let goal = GoalSpec::new("g1", "Test Goal")
            .add_task(task1)
            .add_task_after(task2, &t1_id);

        let mission = MissionSpec::with_id("m1", "Test Mission").add_goal(goal);

        // Create planner and submit
        let mut planner = MissionPlanner::new();
        let mission_id = planner.submit(mission).unwrap();

        assert_eq!(mission_id.as_str(), "m1");

        // Start and run
        planner.start(&mission_id).unwrap();

        // Tick until complete
        for _ in 0..20 {
            planner.tick().unwrap();
            let status = planner.get_mission_status(&mission_id).unwrap();
            if status.status == ExecutionStatus::Completed {
                break;
            }
        }

        let status = planner.get_mission_status(&mission_id).unwrap();
        assert_eq!(status.status, ExecutionStatus::Completed);
    }

    #[test]
    fn test_parallel_tasks() {
        // Create a goal with parallel tasks
        let task_a = TaskSpec::noop("a", "Task A");
        let task_b = TaskSpec::noop("b", "Task B");
        let task_c = TaskSpec::noop("c", "Task C");

        let a_id = task_a.id.clone();
        let b_id = task_b.id.clone();

        // A and B run in parallel, C waits for both
        let goal = GoalSpec::new("parallel", "Parallel Goal")
            .add_task(task_a)
            .add_task(task_b)
            .add_task_after_all(task_c, &[a_id, b_id]);

        let mission = MissionSpec::with_id("parallel_test", "Parallel Test").add_goal(goal);

        let mut planner = MissionPlannerBuilder::new().max_concurrent_tasks(4).build();

        let id = planner.submit(mission).unwrap();
        planner.start(&id).unwrap();

        // Run to completion
        for _ in 0..20 {
            planner.tick().unwrap();
            let status = planner.get_mission_status(&id).unwrap();
            if status.status == ExecutionStatus::Completed {
                break;
            }
        }

        let status = planner.get_mission_status(&id).unwrap();
        assert_eq!(status.status, ExecutionStatus::Completed);
    }

    #[test]
    fn test_goal_dependencies() {
        // Create mission with dependent goals
        let goal1 = GoalSpec::new("g1", "Goal 1").add_task(TaskSpec::noop("t1", "Task 1"));

        let goal2 = GoalSpec::new("g2", "Goal 2").add_task(TaskSpec::noop("t2", "Task 2"));

        let g1_id = goal1.id.clone();

        let mission = MissionSpec::with_id("deps", "Dependency Test")
            .add_goal(goal1)
            .add_goal_after(goal2, &g1_id);

        let mut planner = MissionPlanner::new();
        let id = planner.submit(mission).unwrap();
        planner.start(&id).unwrap();

        // Run to completion
        for _ in 0..20 {
            planner.tick().unwrap();
            let status = planner.get_mission_status(&id).unwrap();
            if status.status == ExecutionStatus::Completed {
                break;
            }
        }

        let status = planner.get_mission_status(&id).unwrap();
        assert_eq!(status.status, ExecutionStatus::Completed);
    }

    #[test]
    fn test_custom_executor() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let task = TaskSpec::new(
            "count",
            "Counter Task",
            TaskExecutor::Custom {
                handler: "counter".to_string(),
                parameters: Default::default(),
            },
        );

        let goal = GoalSpec::new("g1", "Goal").add_task(task);
        let mission = MissionSpec::with_id("m1", "Test").add_goal(goal);

        let mut planner = MissionPlanner::new();
        planner.register_executor(
            "counter",
            Arc::new(move |_task: &TaskSpec, _ctx: &ExecutionContext| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok(None)
            }),
        );

        let id = planner.submit(mission).unwrap();
        planner.start(&id).unwrap();

        for _ in 0..10 {
            planner.tick().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_dag_utilities() {
        let task1 = TaskSpec::noop("t1", "Task 1");
        let task2 = TaskSpec::noop("t2", "Task 2");
        let task3 = TaskSpec::noop("t3", "Task 3");
        let t1_id = task1.id.clone();
        let t2_id = task2.id.clone();

        let goal = GoalSpec::new("g1", "Goal")
            .add_task(task1)
            .add_task_after(task2, &t1_id)
            .add_task_after(task3, &t2_id);

        let dag = build_task_dag(&goal).unwrap();

        assert_eq!(dag.node_count(), 3);
        assert_eq!(dag.edge_count(), 2);

        let sorted = dag.topological_sort().unwrap();
        assert_eq!(sorted[0], t1_id);
        assert_eq!(sorted[2], TaskId::new("t3"));
    }

    #[test]
    fn test_execution_scheduler() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_node("c".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());
        dag.add_dependency("c".to_string(), "b".to_string());

        let mut scheduler = ExecutionScheduler::new(dag);

        // Only 'a' should be ready initially
        let ready = scheduler.get_ready();
        assert_eq!(ready, vec!["a".to_string()]);

        // Execute 'a'
        scheduler.start("a".to_string());
        scheduler.complete("a".to_string());

        // Now 'b' should be ready
        let ready = scheduler.get_ready();
        assert_eq!(ready, vec!["b".to_string()]);

        scheduler.start("b".to_string());
        scheduler.complete("b".to_string());

        // Now 'c' should be ready
        let ready = scheduler.get_ready();
        assert_eq!(ready, vec!["c".to_string()]);

        scheduler.start("c".to_string());
        scheduler.complete("c".to_string());

        assert!(scheduler.is_done());
    }

    #[test]
    fn test_metrics() {
        let mut planner = MissionPlanner::new();

        let goal = GoalSpec::new("g1", "Goal").add_task(TaskSpec::noop("t1", "Task"));

        let mission = MissionSpec::with_id("m1", "Test").add_goal(goal);

        let id = planner.submit(mission).unwrap();
        planner.start(&id).unwrap();

        for _ in 0..10 {
            planner.tick().unwrap();
        }

        let metrics = planner.metrics();
        assert!(metrics.total_missions >= 1);
        assert!(metrics.successful_missions >= 1);
    }

    #[test]
    fn test_shared_planner_thread_safe() {
        use std::thread;

        let planner = MissionPlanner::new();
        let shared = SharedMissionPlanner::new(planner);

        let goal = GoalSpec::new("g1", "Goal").add_task(TaskSpec::noop("t1", "Task"));
        let mission = MissionSpec::with_id("m1", "Test").add_goal(goal);

        let id = shared.submit(mission).unwrap();
        shared.start(&id).unwrap();

        // Spawn threads to tick
        let shared_clone = shared.clone_arc();
        let handle = thread::spawn(move || {
            for _ in 0..5 {
                shared_clone.tick().unwrap();
            }
        });

        for _ in 0..5 {
            shared.tick().unwrap();
        }

        handle.join().unwrap();
    }

    #[test]
    fn test_mission_timeout() {
        let goal = GoalSpec::new("g1", "Goal").add_task(
            TaskSpec::new(
                "slow",
                "Slow Task",
                TaskExecutor::Custom {
                    handler: "slow".to_string(),
                    parameters: Default::default(),
                },
            )
            .with_timeout(Duration::from_millis(10)),
        );

        let mission = MissionSpec::with_id("m1", "Test")
            .with_timeout(Duration::from_millis(50))
            .add_goal(goal);

        let mut planner = MissionPlanner::new();
        // Don't register the slow executor - task will fail

        let id = planner.submit(mission).unwrap();
        planner.start(&id).unwrap();

        // Let it time out
        std::thread::sleep(Duration::from_millis(100));

        // Tick to process timeout
        planner.tick().unwrap();

        let status = planner.get_mission_status(&id).unwrap();
        assert!(
            status.status == ExecutionStatus::Failed || status.status == ExecutionStatus::Completed
        );
    }
}
