---
name: explain-in-simple-english
description: Explain technical topics in English with an approximately 80% ASD-STE100-inspired writing style. Use when the user requests Simplified Technical English, STE-style explanations, or clear English explanations that preserve technical accuracy.
---

# Explain in Simple English

Always write the explanation in English, even when the request
or source material uses another language.

Apply an approximately 80% ASD-STE100-inspired writing style.
Treat "80%" as a qualitative style target, not a measured
compliance score. Do not claim formal ASD-STE100 compliance.

## Writing rules

- Use short, clear sentences with one main idea per sentence.
- Prefer active voice and identify who or what performs each action.
- Use familiar words when they preserve the exact meaning.
- Use one consistent term for each concept.
- Define necessary technical terms at first use.
- Avoid idioms, ambiguous expressions, and unnecessary jargon.
- Make conditions, causes, and consequences explicit.
- Use short paragraphs in a logical order.
- Use lists for steps and tables for comparisons when useful.
- Allow longer sentences when splitting them would obscure meaning.
- Simplify the wording without reducing the requested depth.

## Preserve technical accuracy

- Preserve important facts, conditions, exceptions, and limitations.
- Keep equations, units, code, identifiers, and numerical values accurate.
- Distinguish requirements, recommendations, and possibilities.
- Do not turn conditional statements into absolute claims.
- Explain necessary technical terms instead of removing them.
- Label analogies and explain their limits when relevant.
- State uncertainty when the available information is incomplete.
- Prefer accuracy when a simpler expression would change the meaning.

## Workflow

1. Identify the topic, audience, and requested level of detail.
   Use available context. Ask only when missing information
   would materially change the explanation.
2. State the main idea first.
3. Explain the details in a logical sequence.
4. Add a concrete example when it improves understanding.
5. Review for ambiguity, unnecessary words, and inconsistent terms.
6. Check that the explanation preserves the technical meaning.

Follow the user's requested format and length.
Do not force every explanation into the same template.
Do not include a style-compliance report unless requested.

## Style example

Normal:
The utilization of asynchronous processing mechanisms facilitates
the optimization of application responsiveness by enabling
computational tasks to be executed without unnecessarily
obstructing the primary execution flow.

Preferred:
Asynchronous processing lets a program run tasks without blocking
the main execution flow. This can help the application respond faster.

Use this example to guide wording and sentence structure.
Do not treat it as a universal guarantee about asynchronous processing.