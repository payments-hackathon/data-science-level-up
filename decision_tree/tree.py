# flake8: noqa

# %%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
import random

# Load and prepare data
def load_data():
    # Load training data
    train_df = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')
    
    # Select key features for decision tree
    features = ['TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 
                'TRANSACTION_CASHBACK_AMOUNT', 'CARD_BRAND', 'TRANSACTION_TYPE',
                'TRANSACTION_STATUS', 'TRANSACTION_CURRENCY', 'IS_RECURRING_TRANSACTION']
    
    X = train_df[features].copy()
    y = train_df['TX_FRAUD']
    
    # Encode categorical variables
    le_dict = {}
    for col in X.select_dtypes(include=['object', 'bool']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Individual representation: [max_depth, min_samples_split, min_samples_leaf, criterion]
toolbox.register("max_depth", random.randint, 3, 20)
toolbox.register("min_samples_split", random.randint, 2, 20)
toolbox.register("min_samples_leaf", random.randint, 1, 10)
toolbox.register("criterion", random.choice, [0, 1])  # 0=gini, 1=entropy

toolbox.register("individual", tools.initCycle, creator.Individual,
                (toolbox.max_depth, toolbox.min_samples_split, 
                 toolbox.min_samples_leaf, toolbox.criterion), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def evaluate_tree(individual, X_train, X_val, y_train, y_val):
    max_depth, min_samples_split, min_samples_leaf, criterion_idx = individual
    criterion = 'gini' if criterion_idx == 0 else 'entropy'
    
    try:
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        return (score,)
    except:
        return (0.0,)

# Genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[3, 2, 1, 0], up=[20, 20, 10, 1], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def evolve_decision_tree():
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    
    # Register evaluation function with data
    toolbox.register("evaluate", evaluate_tree, X_train=X_train, X_val=X_val, 
                    y_train=y_train, y_val=y_val)
    
    # Evolution parameters
    pop_size = 50
    generations = 20
    cx_prob = 0.7
    mut_prob = 0.2
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Evolution
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cx_prob, mut_prob, generations, 
        stats=stats, verbose=True
    )
    
    # Best individual
    best_ind = tools.selBest(pop, 1)[0]
    max_depth, min_samples_split, min_samples_leaf, criterion_idx = best_ind
    criterion = 'gini' if criterion_idx == 0 else 'entropy'
    
    print(f"\nBest parameters:")
    print(f"Max depth: {max_depth}")
    print(f"Min samples split: {min_samples_split}")
    print(f"Min samples leaf: {min_samples_leaf}")
    print(f"Criterion: {criterion}")
    print(f"Best fitness (ROC-AUC): {best_ind.fitness.values[0]:.4f}")
    
    # Train final model
    best_clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    
    # Combine train and validation for final training
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    best_clf.fit(X_full, y_full)
    
    return best_clf, best_ind.fitness.values[0]

random.seed(42)
np.random.seed(42)

print("Evolving Decision Tree Classifier...")
best_model, best_score = evolve_decision_tree()
print(f"\nEvolution complete. Best ROC-AUC: {best_score:.4f}")

# %%
import joblib

# Save the evolved model
joblib.dump(best_model, 'evolved_decision_tree.pkl')
print("Model saved as 'evolved_decision_tree.pkl'")