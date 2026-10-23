---

name: self-documenting-code
description: Write, refactor, and review code so that its intent, domain meaning,
  decisions, and data flow are clear from the code itself. Use when creating
  or modifying code where readability and maintainability matter, especially
  when code contains complex expressions, Boolean conditions, vague names,
  magic values, deep nesting, dense transformations, or comments that merely
  restate implementation. Prefer intent-revealing names, meaningful
  intermediate variables as executable documentation, explicit conceptual
  steps, guard clauses, and domain-level abstractions when justified.
  Avoid meaningless temporary variables, variable explosion, clever
  one-liners, redundant comments, and unnecessary abstractions.

---

# Self-Documenting Code

## Core Principles

### 1. Code Should Reveal Intent

Names should describe **meaning and intent**, not merely implementation details.

Avoid vague names such as:

```
data
value
result
temp
obj
item
flag
x
ret
```

Prefer names such as:

```
active_users
remaining_capacity
retry_delay
shipping_cost
has_permission
should_retry
is_subscription_active
```

A reader should understand the role of a value without tracing where it came from.

---

### 2. Use Intermediate Variables as Documentation

Use meaningful intermediate variables to explain non-obvious expressions.

An intermediate variable is useful when its name provides information that the raw expression does not communicate clearly.

### Avoid

```
if response.status_code >= 500 and retry_count < 3 and time.time() - last_retry > 2:
    retry()
```

### Prefer

```
is_server_error = response.status_code >= 500
has_retries_remaining = retry_count < MAX_RETRIES
retry_delay_elapsed = time.time() - last_retry > RETRY_DELAY_SECONDS

should_retry = (
    is_server_error
    and has_retries_remaining
    and retry_delay_elapsed
)

if should_retry:
    retry()
```

The intermediate variables act as executable documentation.

---

## 3. Name Meaning, Not Mechanics

Translate low-level implementation into domain language.

### Avoid

```
temperature > 80
```

when the actual meaning is:

```
is_gpu_overheating
```

Prefer:

```
is_gpu_overheating = temperature > GPU_THERMAL_LIMIT
```

And when the concept belongs to the domain model, consider:

```
gpu.is_overheating()
```

Use this progression when appropriate:

```
raw implementation
        ↓
meaningful variable
        ↓
domain-level operation
```

---

## 4. One Conceptual Step Per Statement

Each statement should represent one clear idea.

This does **not** mean that a statement may contain only one CPU operation.

For example:

```
total_price = unit_price * quantity
```

is one conceptual step:

```
calculate total price
```

That is acceptable.

Avoid statements that force the reader to understand several unrelated transformations simultaneously.

### Avoid

```
users = sorted(
    [u for u in db.query(User).filter(User.active == True) if u.age >= 18],
    key=lambda u: u.created_at,
    reverse=True,
)[:10]
```

### Prefer

```
active_users = db.query(User).filter(User.active == True)

adult_users = [
    user
    for user in active_users
    if user.age >= 18
]

newest_users_first = sorted(
    adult_users,
    key=lambda user: user.created_at,
    reverse=True,
)

top_ten_users = newest_users_first[:10]
```

Each step should communicate one meaningful transformation.

---

## 5. Use Semantic Boolean Variables

Complex Boolean expressions should often be decomposed into named Boolean concepts.

Prefer names beginning with forms such as:

```
is_
has_
can_
should_
needs_
was_
contains_
supports_
allows_
```

### Avoid

```
if ((user.age >= 18 && user.account != nullptr) &&
    (user.account->status == ACTIVE ||
     user.account->trialEnd > now) &&
    !user.isBlocked) {
    grantAccess();
}
```

### Prefer

```
const bool isAdult = user.age >= 18;
const bool hasAccount = user.account != nullptr;

const bool hasActiveAccount =
    hasAccount && user.account->status == ACTIVE;

const bool hasActiveTrial =
    hasAccount && user.account->trialEnd > now;

const bool hasAccessEntitlement =
    hasActiveAccount || hasActiveTrial;

const bool canAccess =
    isAdult &&
    hasAccessEntitlement &&
    !user.isBlocked;

if (canAccess) {
    grantAccess();
}
```

The top-level condition should communicate the decision being made.

---

## 6. Extract Business Meaning from Expressions

Raw expressions often describe mechanics rather than meaning.

Convert mechanics into business concepts.

### Avoid

```
if (route.distance < 20) {
    return SAME_DAY;
}
```

### Prefer

```
const isEligibleForSameDayDelivery =
    route.distance < SAME_DAY_DISTANCE_LIMIT;

if (isEligibleForSameDayDelivery) {
    return SAME_DAY;
}
```

The variable name should explain **why the expression matters**.

---

## 7. Replace Magic Values with Named Concepts

Avoid unexplained numeric or string literals when the value represents a policy, threshold, configuration, protocol value, or domain rule.

### Avoid

```
if retry_count < 3:
```

Prefer:

```
if retry_count < MAX_RETRIES:
```

### Avoid

```
if route.distance < 20:
```

Prefer:

```
if route.distance < SAME_DAY_DISTANCE_LIMIT:
```

Constants should explain the meaning of the value.

---

## 8. Do Not Introduce Meaningless Intermediate Variables

Intermediate variables are documentation only when their names add semantic information.

### Avoid

```
name = user.get_name()
return name
```

Prefer:

```
return user.get_name()
```

Also avoid:

```
value1 = quantity * price
value2 = value1 * 0.1
value3 = min(value2, 100)
```

Prefer:

```
base_price = quantity * unit_price
calculated_shipping = base_price * SHIPPING_RATE
shipping_cost = min(calculated_shipping, MAX_SHIPPING_COST)
```

Use this test:

> Does the variable name tell the reader something meaningful that the expression itself does not clearly communicate?

If not, keep the expression inline.

---

## 9. Prefer Stable Meaning Per Variable

A variable should represent one concept throughout its lifetime.

Avoid reusing a variable for unrelated meanings.

### Avoid

```
result = load_user()
result = calculate_discount(result)
result = serialize(result)
```

Prefer:

```
user = load_user()
discount = calculate_discount(user)
serialized_discount = serialize(discount)
```

Avoid making the reader continuously reinterpret what a variable represents.

---

## 10. Functions Should Read Like a Story

Top-level functions should communicate workflow rather than implementation details.

Prefer code such as:

```
validate_request()
customer = load_customer()
price = calculate_price(customer)
discounted_price = apply_discount(price)
save_order(customer, discounted_price)
send_confirmation(customer)
```

The reader should understand the algorithm before entering helper functions.

Aim for:

```
validate
→ load
→ calculate
→ apply
→ persist
→ notify
```

---

## 11. Extract Functions When the Concept Is Larger Than a Value

Use an intermediate variable for a meaningful expression.

Use a function when a meaningful concept contains multiple operations or is reusable.

Example:

```
is_high_value_order = order.total > HIGH_VALUE_THRESHOLD
```

is suitable when used locally.

If the concept is reused or belongs to the domain model, prefer:

```
order.is_high_value()
```

Choose the abstraction level based on scope:

```
expression once
→ intermediate variable

logic within one function
→ local helper when useful

logic reused across functions
→ function or method

important domain rule
→ domain method or domain type
```

---

## 12. Prefer Guard Clauses Over Deep Nesting

Keep the main path visually obvious.

### Avoid

```
if user is not None:
    if user.active:
        if user.has_permission:
            process(user)
```

### Prefer

```
if user is None:
    return

if not user.active:
    return

if not user.has_permission:
    return

process(user)
```

Or, when domain meaning benefits from naming:

```
is_eligible = (
    user.active
    and user.has_permission
)

if not is_eligible:
    return

process(user)
```

Choose whichever version communicates intent more clearly.

---

## 13. Comments Should Explain Why

Do not use comments to restate obvious code.

### Avoid

```
# Increment retry count
retry_count += 1
```

Prefer comments for:

- external system limitations,
    
- unusual business rules,
    
- compatibility requirements,
    
- performance tradeoffs,
    
- security constraints,
    
- workarounds,
    
- architectural decisions,
    
- non-obvious reasons.
    

Example:

```
# The vendor starts aggressive throttling after three
# consecutive retries, so retries are intentionally capped.
MAX_RETRIES = 3
```

Code should normally explain **what**.

Comments should explain **why**.

---

## 14. Do Not Over-Abstract

Self-documenting code does not mean creating a function, class, interface, helper, or abstraction for every operation.

Do not replace simple readable code with unnecessary indirection.

### Avoid

```
def get_user_name(user):
    return user.name
```

when:

```
user.name
```

is already clear.

Do not introduce abstractions unless they improve one or more of:

- meaning,
    
- reuse,
    
- separation of concerns,
    
- testability,
    
- domain modeling,
    
- complexity management.
    

---

## 15. Prefer Explicit Data Flow

Make transformations visible.

### Avoid

```
return send(format(validate(parse(request.body))))
```

Prefer:

```
payload = parse(request.body)
validated_payload = validate(payload)
formatted_response = format(validated_payload)

return send(formatted_response)
```

Use intermediate variables when they make the transformation pipeline easier to inspect, debug, test, or understand.

---

## 16. Avoid Clever Compression

Do not optimize for minimum line count.

Readable code may be longer than equivalent compressed code.

Prefer:

```
base_price = order.quantity * order.unit_price

discountable_quantity = max(
    0,
    order.quantity - DISCOUNT_THRESHOLD,
)

quantity_discount = (
    discountable_quantity
    * order.unit_price
    * DISCOUNT_RATE
)

shipping_cost = min(
    base_price * SHIPPING_RATE,
    MAX_SHIPPING_COST,
)

final_price = (
    base_price
    - quantity_discount
    + shipping_cost
)
```

over a single large expression.

Optimize for the reader, not the number of lines.

---

# Refactoring Procedure

When reviewing existing code, use the following sequence.

## Step 1 — Identify Mental Decoding

Look for expressions where the reader has to mentally translate implementation into meaning.

Examples:

```
age >= 18
expires_at > now
status >= 500
count < 3
distance < 20
```

Ask:

> What does this expression mean in the domain?

---

## Step 2 — Name the Meaning

Introduce a semantic name when useful.

Example:

```
has_active_subscription = subscription.expires_at > now
```

Do not merely rename syntax.

Bad:

```
expiration_check = subscription.expires_at > now
```

Better:

```
has_active_subscription = subscription.expires_at > now
```

---

## Step 3 — Decompose Complex Expressions

For expressions containing several independent concepts, extract each meaningful part.

Example:

```
is_server_error = response.status_code >= 500
has_retries_remaining = retry_count < MAX_RETRIES
retry_delay_elapsed = elapsed_time >= RETRY_DELAY

should_retry = (
    is_server_error
    and has_retries_remaining
    and retry_delay_elapsed
)
```

---

## Step 4 — Raise the Abstraction Level

If a concept represents a stable domain rule, consider moving it behind a function or method.

From:

```
is_subscription_active = subscription.expires_at > now
```

to:

```
subscription.is_active(now)
```

only when the abstraction genuinely improves the design.

---

## Step 5 — Remove Redundant Variables

After refactoring, inspect every intermediate variable.

Inline it if:

- the name adds no meaning,
    
- the expression is already obvious,
    
- the variable is used only to immediately return the same value,
    
- or the additional name increases noise.
    

---

# AI-Generated Code Rules

When generating code as an AI agent:

1. Do not optimize for the shortest possible implementation.
    
2. Do not produce dense one-liners unless they are obviously clearer.
    
3. Do not introduce `temp`, `result`, `data`, or numbered variables without a specific semantic reason.
    
4. Decompose non-trivial Boolean conditions.
    
5. Prefer named intermediate values for important domain concepts.
    
6. Avoid excessive helper functions that merely wrap one obvious expression.
    
7. Avoid excessive abstraction layers.
    
8. Preserve straightforward data flow.
    
9. Prefer domain terminology already used by the project.
    
10. Follow existing project naming conventions unless they reduce clarity.
    
11. Do not rewrite clear existing code merely to satisfy this skill mechanically.
    
12. Refactor only where readability or maintainability meaningfully improves.
    

---

# Anti-Patterns

## Anti-Pattern: Variable Explosion

Do not turn every expression into a variable.

Bad:

```
first_name = user.first_name
last_name = user.last_name
full_name = first_name + " " + last_name

return full_name
```

If no additional meaning is gained:

```
return f"{user.first_name} {user.last_name}"
```

may be clearer.

---

## Anti-Pattern: Fake Semantic Names

Bad:

```
check1 = user.age >= 18
check2 = user.active
final_check = check1 and check2
```

Better:

```
is_adult = user.age >= 18
is_active = user.active

can_register = is_adult and is_active
```

---

## Anti-Pattern: Comment-Driven Code

Bad:

```
# Check whether user has permission
if user.role == "admin" or user.role == "manager":
```

Better:

```
has_management_access = user.role in MANAGEMENT_ROLES

if has_management_access:
```

---

## Anti-Pattern: Premature Abstraction

Bad:

```
class AdultEligibilityEvaluator:
    def evaluate(self, user):
        return user.age >= 18
```

when:

```
is_adult = user.age >= 18
```

is sufficient.

---

## Anti-Pattern: Hidden Side Effects in Expressions

Avoid combining mutation, validation, transformation, and external calls into one expression.

Prefer making important side effects explicit.

Bad:

```
if cache.set(key, fetch_and_transform()) and metrics.increment("success"):
    return True
```

Better:

```
raw_data = fetch_data()
transformed_data = transform(raw_data)

cache_updated = cache.set(key, transformed_data)

if not cache_updated:
    return False

metrics.increment("success")

return True
```

---

# Review Checklist

Before considering code complete, ask:

### Naming

- Do names express intent rather than mechanics?
    
- Are vague names such as `data`, `temp`, `value`, `result`, or `flag` avoidable?
    
- Do Boolean variables read naturally as propositions?
    

### Expressions

- Does any expression require mental decoding?
    
- Would a meaningful intermediate variable improve understanding?
    
- Are complex Boolean expressions decomposed appropriately?
    

### Flow

- Can the main function be understood without reading every implementation detail?
    
- Does each statement represent one clear conceptual step?
    
- Is important data transformation visible?
    

### Abstraction

- Should any repeated or important domain concept become a function or method?
    
- Has unnecessary abstraction been introduced?
    
- Are there trivial helper functions that should be inlined?
    

### Comments

- Are comments explaining why rather than repeating what?
    
- Could a comment be replaced with a better name?
    
- Are important external constraints documented?
    

### Noise

- Does every intermediate variable add meaning?
    
- Can any redundant intermediate variable be safely removed?
    
- Is the code longer because it is clearer, or merely more verbose?
    

---

# Decision Rule for Intermediate Variables

Before extracting an expression into a variable, ask:

> Does the proposed variable name communicate meaningful information that is not immediately obvious from the expression?

If yes, extract it.

If no, keep the expression inline.

Example:

```
has_active_subscription = subscription.expires_at > now
```

adds meaning.

But:

```
user_name = user.name
```

usually does not.

---

# Preferred Style

The preferred transformation is:

```
implementation detail
        ↓
semantic intermediate value
        ↓
intent-revealing decision
        ↓
domain abstraction when justified
```

Example:

```
elapsed_time = current_time - last_retry_time

retry_delay_elapsed = (
    elapsed_time >= RETRY_DELAY_SECONDS
)

has_retries_remaining = (
    retry_count < MAX_RETRIES
)

is_retryable_error = (
    response.status_code >= 500
)

should_retry = (
    is_retryable_error
    and has_retries_remaining
    and retry_delay_elapsed
)

if should_retry:
    retry_request()
```

The final code should read as an explanation of the algorithm.

---

# Final Principle

Write code for the next human reader.

Use names, intermediate variables, functions, constants, and domain abstractions to transform low-level implementation into understandable intent.

The goal is not:

```
more variables
more functions
more abstractions
fewer comments
```

The goal is:

```
less mental decoding
clearer intent
visible data flow
meaningful domain language
minimal unnecessary complexity
```

Prefer code that explains itself.