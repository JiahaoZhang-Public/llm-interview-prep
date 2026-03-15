# Flashcards

## What is the minimum loop for tool calling?

Model decides tool -> tool executes -> observation returns to the model.

## Why parse function schemas strictly?

Loose parsing leads to brittle tool execution and silent failures.

## What belongs in short-term memory?

Only the local task state that should influence the next tool or reasoning step.

## What makes ReAct debuggable?

Thought, action, and observation are explicit and inspectable.
