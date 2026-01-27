//! Task DAG (Directed Acyclic Graph) Implementation
//!
//! This module provides dependency graph management for tasks and goals,
//! including topological sorting, cycle detection, and execution order.

use std::collections::{HashMap, HashSet, VecDeque};

use super::types::{ExecutionStatus, GoalId, GoalSpec, MissionPlannerError, MissionSpec, TaskId};

/// A directed acyclic graph for managing task/goal dependencies.
///
/// The DAG tracks which items depend on which other items and provides
/// methods for determining execution order, detecting cycles, and
/// finding ready-to-execute items.
#[derive(Debug, Clone)]
pub struct DependencyGraph<Id: std::hash::Hash + Eq + Clone> {
    /// Map from node to its dependencies (what it depends on).
    dependencies: HashMap<Id, HashSet<Id>>,
    /// Map from node to its dependents (what depends on it).
    dependents: HashMap<Id, HashSet<Id>>,
    /// All nodes in the graph.
    nodes: HashSet<Id>,
}

impl<Id: std::hash::Hash + Eq + Clone + std::fmt::Display> DependencyGraph<Id> {
    /// Create a new empty dependency graph.
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, id: Id) {
        self.nodes.insert(id.clone());
        self.dependencies.entry(id.clone()).or_default();
        self.dependents.entry(id).or_default();
    }

    /// Add a dependency: `from` depends on `to`.
    pub fn add_dependency(&mut self, from: Id, to: Id) {
        self.add_node(from.clone());
        self.add_node(to.clone());

        self.dependencies
            .entry(from.clone())
            .or_default()
            .insert(to.clone());
        self.dependents.entry(to).or_default().insert(from);
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: &Id) {
        // Remove from dependencies of other nodes
        if let Some(deps) = self.dependencies.remove(id) {
            for dep in deps {
                if let Some(dependents) = self.dependents.get_mut(&dep) {
                    dependents.remove(id);
                }
            }
        }

        // Remove from dependents of other nodes
        if let Some(deps) = self.dependents.remove(id) {
            for dep in deps {
                if let Some(dependencies) = self.dependencies.get_mut(&dep) {
                    dependencies.remove(id);
                }
            }
        }

        self.nodes.remove(id);
    }

    /// Get the dependencies of a node (what it depends on).
    pub fn get_dependencies(&self, id: &Id) -> Option<&HashSet<Id>> {
        self.dependencies.get(id)
    }

    /// Get the dependents of a node (what depends on it).
    pub fn get_dependents(&self, id: &Id) -> Option<&HashSet<Id>> {
        self.dependents.get(id)
    }

    /// Check if a node has any unmet dependencies given a set of completed nodes.
    pub fn has_unmet_dependencies(&self, id: &Id, completed: &HashSet<Id>) -> bool {
        self.dependencies
            .get(id)
            .is_some_and(|deps| !deps.is_subset(completed))
    }

    /// Get nodes that are ready to execute (no unmet dependencies).
    pub fn get_ready_nodes(&self, completed: &HashSet<Id>) -> Vec<Id> {
        self.nodes
            .iter()
            .filter(|id| !completed.contains(*id))
            .filter(|id| !self.has_unmet_dependencies(id, completed))
            .cloned()
            .collect()
    }

    /// Get nodes with no dependencies (entry points).
    pub fn get_entry_nodes(&self) -> Vec<Id> {
        self.nodes
            .iter()
            .filter(|id| {
                self.dependencies
                    .get(*id)
                    .is_none_or(|deps| deps.is_empty())
            })
            .cloned()
            .collect()
    }

    /// Get nodes with no dependents (exit points).
    pub fn get_exit_nodes(&self) -> Vec<Id> {
        self.nodes
            .iter()
            .filter(|id| self.dependents.get(*id).is_none_or(|deps| deps.is_empty()))
            .cloned()
            .collect()
    }

    /// Perform topological sort using Kahn's algorithm.
    ///
    /// Returns nodes in execution order, or an error if a cycle is detected.
    pub fn topological_sort(&self) -> Result<Vec<Id>, MissionPlannerError> {
        let mut in_degree: HashMap<Id, usize> = self
            .nodes
            .iter()
            .map(|id| {
                let degree = self.dependencies.get(id).map_or(0, |deps| deps.len());
                (id.clone(), degree)
            })
            .collect();

        let mut queue: VecDeque<Id> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut result = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(dependents) = self.dependents.get(&node) {
                for dependent in dependents {
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent.clone());
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            // Find nodes involved in the cycle
            let remaining: Vec<_> = self
                .nodes
                .iter()
                .filter(|id| !result.contains(id))
                .map(|id| id.to_string())
                .collect();
            return Err(MissionPlannerError::CyclicDependency(remaining.join(", ")));
        }

        Ok(result)
    }

    /// Check if the graph has any cycles.
    pub fn has_cycle(&self) -> bool {
        self.topological_sort().is_err()
    }

    /// Validate the graph for cycles and invalid dependencies.
    pub fn validate(&self) -> Result<(), MissionPlannerError> {
        // Check for cycles
        self.topological_sort()?;

        // Check for self-dependencies
        for (id, deps) in &self.dependencies {
            if deps.contains(id) {
                return Err(MissionPlannerError::CyclicDependency(format!(
                    "Self-dependency: {}",
                    id
                )));
            }
        }

        Ok(())
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges (dependencies) in the graph.
    pub fn edge_count(&self) -> usize {
        self.dependencies.values().map(|deps| deps.len()).sum()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &HashSet<Id> {
        &self.nodes
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.dependencies.clear();
        self.dependents.clear();
        self.nodes.clear();
    }

    /// Get the transitive dependencies of a node (all nodes it depends on, recursively).
    pub fn transitive_dependencies(&self, id: &Id) -> HashSet<Id> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(deps) = self.dependencies.get(id) {
            queue.extend(deps.iter().cloned());
        }

        while let Some(dep) = queue.pop_front() {
            if result.insert(dep.clone()) {
                if let Some(deps) = self.dependencies.get(&dep) {
                    queue.extend(deps.iter().cloned());
                }
            }
        }

        result
    }

    /// Get the transitive dependents of a node (all nodes that depend on it, recursively).
    pub fn transitive_dependents(&self, id: &Id) -> HashSet<Id> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(deps) = self.dependents.get(id) {
            queue.extend(deps.iter().cloned());
        }

        while let Some(dep) = queue.pop_front() {
            if result.insert(dep.clone()) {
                if let Some(deps) = self.dependents.get(&dep) {
                    queue.extend(deps.iter().cloned());
                }
            }
        }

        result
    }

    /// Get the critical path length (longest dependency chain).
    pub fn critical_path_length(&self) -> usize {
        let sorted = match self.topological_sort() {
            Ok(s) => s,
            Err(_) => return 0,
        };

        let mut distances: HashMap<Id, usize> = HashMap::new();

        for node in sorted {
            let max_dep_dist = self.dependencies.get(&node).map_or(0, |deps| {
                deps.iter()
                    .map(|d| distances.get(d).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0)
            });
            distances.insert(node, max_dep_dist + 1);
        }

        distances.values().copied().max().unwrap_or(0)
    }
}

impl<Id: std::hash::Hash + Eq + Clone + std::fmt::Display> Default for DependencyGraph<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// Task DAG for managing task execution order within a goal.
pub type TaskDAG = DependencyGraph<TaskId>;

/// Goal DAG for managing goal execution order within a mission.
pub type GoalDAG = DependencyGraph<GoalId>;

/// Build a task DAG from a goal specification.
pub fn build_task_dag(goal: &GoalSpec) -> Result<TaskDAG, MissionPlannerError> {
    let mut dag = TaskDAG::new();

    // Add all tasks
    for task in &goal.tasks {
        dag.add_node(task.id.clone());
    }

    // Add dependencies
    for (task_id, deps) in &goal.dependencies {
        for dep in deps {
            // Validate that the dependency exists
            if !goal.tasks.iter().any(|t| &t.id == dep) {
                return Err(MissionPlannerError::InvalidDependency {
                    from: task_id.to_string(),
                    to: dep.to_string(),
                });
            }
            dag.add_dependency(task_id.clone(), dep.clone());
        }
    }

    // Validate the DAG
    dag.validate()?;

    Ok(dag)
}

/// Build a goal DAG from a mission specification.
pub fn build_goal_dag(mission: &MissionSpec) -> Result<GoalDAG, MissionPlannerError> {
    let mut dag = GoalDAG::new();

    // Add all goals
    for goal in &mission.goals {
        dag.add_node(goal.id.clone());
    }

    // Add dependencies
    for (goal_id, deps) in &mission.dependencies {
        for dep in deps {
            // Validate that the dependency exists
            if !mission.goals.iter().any(|g| &g.id == dep) {
                return Err(MissionPlannerError::InvalidDependency {
                    from: goal_id.to_string(),
                    to: dep.to_string(),
                });
            }
            dag.add_dependency(goal_id.clone(), dep.clone());
        }
    }

    // Validate the DAG
    dag.validate()?;

    Ok(dag)
}

/// Execution scheduler that uses DAGs to determine what can run next.
pub struct ExecutionScheduler<Id: std::hash::Hash + Eq + Clone + std::fmt::Display> {
    dag: DependencyGraph<Id>,
    completed: HashSet<Id>,
    failed: HashSet<Id>,
    running: HashSet<Id>,
    skipped: HashSet<Id>,
}

impl<Id: std::hash::Hash + Eq + Clone + std::fmt::Display> ExecutionScheduler<Id> {
    /// Create a new execution scheduler from a DAG.
    pub fn new(dag: DependencyGraph<Id>) -> Self {
        Self {
            dag,
            completed: HashSet::new(),
            failed: HashSet::new(),
            running: HashSet::new(),
            skipped: HashSet::new(),
        }
    }

    /// Mark a node as started.
    pub fn start(&mut self, id: Id) {
        self.running.insert(id);
    }

    /// Mark a node as completed successfully.
    pub fn complete(&mut self, id: Id) {
        self.running.remove(&id);
        self.completed.insert(id);
    }

    /// Mark a node as failed.
    pub fn fail(&mut self, id: Id) {
        self.running.remove(&id);
        self.failed.insert(id);
    }

    /// Mark a node as skipped.
    pub fn skip(&mut self, id: Id) {
        self.running.remove(&id);
        self.skipped.insert(id);
    }

    /// Reset a node for retry - removes from running/failed so it becomes ready again.
    pub fn reset_for_retry(&mut self, id: Id) {
        self.running.remove(&id);
        self.failed.remove(&id);
    }

    /// Get nodes that are ready to execute.
    ///
    /// A node is ready if:
    /// - It hasn't been completed, failed, skipped, or is running
    /// - All its dependencies are complete (not failed or skipped)
    pub fn get_ready(&self) -> Vec<Id> {
        let blocked: HashSet<_> = self
            .completed
            .iter()
            .chain(self.failed.iter())
            .chain(self.skipped.iter())
            .chain(self.running.iter())
            .cloned()
            .collect();

        self.dag
            .nodes()
            .iter()
            .filter(|id| !blocked.contains(*id))
            .filter(|id| !self.dag.has_unmet_dependencies(id, &self.completed))
            .cloned()
            .collect()
    }

    /// Get nodes that are blocked due to failed dependencies.
    pub fn get_blocked_by_failure(&self) -> Vec<Id> {
        let done: HashSet<_> = self
            .completed
            .iter()
            .chain(self.failed.iter())
            .chain(self.skipped.iter())
            .chain(self.running.iter())
            .cloned()
            .collect();

        self.dag
            .nodes()
            .iter()
            .filter(|id| !done.contains(*id))
            .filter(|id| {
                // Check if any dependency failed
                self.dag
                    .get_dependencies(id)
                    .is_some_and(|deps| deps.iter().any(|d| self.failed.contains(d)))
            })
            .cloned()
            .collect()
    }

    /// Check if all nodes are done (completed, failed, or skipped).
    pub fn is_done(&self) -> bool {
        let done_count = self.completed.len() + self.failed.len() + self.skipped.len();
        done_count == self.dag.node_count()
    }

    /// Check if there are any running nodes.
    pub fn has_running(&self) -> bool {
        !self.running.is_empty()
    }

    /// Get progress (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        let total = self.dag.node_count();
        if total == 0 {
            return 1.0;
        }
        let done = self.completed.len() + self.failed.len() + self.skipped.len();
        done as f64 / total as f64
    }

    /// Get the status of a specific node.
    pub fn status(&self, id: &Id) -> ExecutionStatus {
        if self.completed.contains(id) {
            ExecutionStatus::Completed
        } else if self.failed.contains(id) {
            ExecutionStatus::Failed
        } else if self.skipped.contains(id) {
            ExecutionStatus::Skipped
        } else if self.running.contains(id) {
            ExecutionStatus::Running
        } else if self.dag.has_unmet_dependencies(id, &self.completed) {
            ExecutionStatus::Blocked
        } else {
            ExecutionStatus::Ready
        }
    }

    /// Get the completed nodes.
    pub fn completed(&self) -> &HashSet<Id> {
        &self.completed
    }

    /// Get the failed nodes.
    pub fn failed(&self) -> &HashSet<Id> {
        &self.failed
    }

    /// Get the running nodes.
    pub fn running(&self) -> &HashSet<Id> {
        &self.running
    }

    /// Get the skipped nodes.
    pub fn skipped(&self) -> &HashSet<Id> {
        &self.skipped
    }

    /// Reset the scheduler state.
    pub fn reset(&mut self) {
        self.completed.clear();
        self.failed.clear();
        self.running.clear();
        self.skipped.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_graph_basic() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());

        assert_eq!(dag.node_count(), 2);
        assert_eq!(dag.edge_count(), 1);
        assert!(dag
            .get_dependencies(&"b".to_string())
            .unwrap()
            .contains(&"a".to_string()));
    }

    #[test]
    fn test_topological_sort() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_node("c".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());
        dag.add_dependency("c".to_string(), "b".to_string());

        let sorted = dag.topological_sort().unwrap();
        assert_eq!(
            sorted,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_dependency("a".to_string(), "b".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());

        assert!(dag.has_cycle());
        assert!(dag.topological_sort().is_err());
    }

    #[test]
    fn test_entry_exit_nodes() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_node("c".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());
        dag.add_dependency("c".to_string(), "b".to_string());

        let entries = dag.get_entry_nodes();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains(&"a".to_string()));

        let exits = dag.get_exit_nodes();
        assert_eq!(exits.len(), 1);
        assert!(exits.contains(&"c".to_string()));
    }

    #[test]
    fn test_ready_nodes() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_node("c".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());
        dag.add_dependency("c".to_string(), "a".to_string());

        let mut completed = HashSet::new();
        let ready = dag.get_ready_nodes(&completed);
        assert_eq!(ready.len(), 1);
        assert!(ready.contains(&"a".to_string()));

        completed.insert("a".to_string());
        let ready = dag.get_ready_nodes(&completed);
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&"b".to_string()));
        assert!(ready.contains(&"c".to_string()));
    }

    #[test]
    fn test_transitive_dependencies() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_node("c".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());
        dag.add_dependency("c".to_string(), "b".to_string());

        let trans = dag.transitive_dependencies(&"c".to_string());
        assert_eq!(trans.len(), 2);
        assert!(trans.contains(&"a".to_string()));
        assert!(trans.contains(&"b".to_string()));
    }

    #[test]
    fn test_critical_path() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_node("c".to_string());
        dag.add_node("d".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());
        dag.add_dependency("c".to_string(), "a".to_string());
        dag.add_dependency("d".to_string(), "b".to_string());
        dag.add_dependency("d".to_string(), "c".to_string());

        // Path a -> b -> d or a -> c -> d, both have length 3
        assert_eq!(dag.critical_path_length(), 3);
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

        // Initially, only 'a' is ready
        let ready = scheduler.get_ready();
        assert_eq!(ready.len(), 1);
        assert!(ready.contains(&"a".to_string()));

        // Start and complete 'a'
        scheduler.start("a".to_string());
        assert!(scheduler.has_running());
        scheduler.complete("a".to_string());

        // Now 'b' is ready
        let ready = scheduler.get_ready();
        assert_eq!(ready.len(), 1);
        assert!(ready.contains(&"b".to_string()));

        scheduler.start("b".to_string());
        scheduler.complete("b".to_string());

        // Now 'c' is ready
        let ready = scheduler.get_ready();
        assert_eq!(ready.len(), 1);
        assert!(ready.contains(&"c".to_string()));

        scheduler.start("c".to_string());
        scheduler.complete("c".to_string());

        assert!(scheduler.is_done());
        assert_eq!(scheduler.progress(), 1.0);
    }

    #[test]
    fn test_scheduler_failure() {
        let mut dag = DependencyGraph::<String>::new();
        dag.add_node("a".to_string());
        dag.add_node("b".to_string());
        dag.add_dependency("b".to_string(), "a".to_string());

        let mut scheduler = ExecutionScheduler::new(dag);

        scheduler.start("a".to_string());
        scheduler.fail("a".to_string());

        // 'b' is blocked by failure
        let blocked = scheduler.get_blocked_by_failure();
        assert_eq!(blocked.len(), 1);
        assert!(blocked.contains(&"b".to_string()));

        // Nothing is ready
        assert!(scheduler.get_ready().is_empty());
    }

    #[test]
    fn test_build_task_dag() {
        use crate::mission_planner::types::{TaskExecutor, TaskSpec};

        let t1 = TaskSpec::noop("t1", "Task 1");
        let t2 = TaskSpec::noop("t2", "Task 2");
        let t1_id = t1.id.clone();

        let goal = GoalSpec::new("g1", "Goal 1")
            .add_task(t1)
            .add_task_after(t2, &t1_id);

        let dag = build_task_dag(&goal).unwrap();
        assert_eq!(dag.node_count(), 2);
        assert_eq!(dag.edge_count(), 1);
    }

    #[test]
    fn test_build_goal_dag() {
        let g1 = GoalSpec::new("g1", "Goal 1");
        let g2 = GoalSpec::new("g2", "Goal 2");
        let g1_id = g1.id.clone();

        let mission = MissionSpec::new("Test")
            .add_goal(g1)
            .add_goal_after(g2, &g1_id);

        let dag = build_goal_dag(&mission).unwrap();
        assert_eq!(dag.node_count(), 2);
        assert_eq!(dag.edge_count(), 1);
    }
}
