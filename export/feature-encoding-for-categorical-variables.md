# Feature encoding for categorical variables

This is an important phase in the data cleaning process. Feature encoding involves replacing classes in a categorical variable with real numbers. For example, classroom A, classroom B and classroom C could be encoded as 2,4,6.

**When should we encode our features? Why?**

We should always encode categorical features so that they can be processed by numerical algorithms, and so machine learning algorithms can learn from them.

**Methods to encode categorical data:**

1. Label encoding (non-ordinal)

Each category is assigned a numeric value not representing any order. For example: [black, red, white] could be encoded to [3,7,11].

2. Label encoding (ordinal)

Each category is assigned a numeric value representing an order. For example: [small, medium, large, extra-large] could be encoded to [1,2,3,4].

3. One hot encoding (binary encoding)

Each category is transformed into a new binary feature, with all records being marked 1 for True or 0 for False. For example: [Florida, Virginia, Massachussets] could be encoded to state_Florida = [1,0,0], state_Virginia = [0,1,0], state_Massachussets = [0,0,1].

