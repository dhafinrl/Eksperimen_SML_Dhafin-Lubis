import json

with open('Eksperimen_Dhafin-Lubis.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# **6. Model Training dan MLflow Tracking**\n", "Melatih model Random Forest dan melakukan tracking eksperimen menggunakan MLflow."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import mlflow\n",
            "import mlflow.sklearn\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import accuracy_score\n",
            "\n",
            "# Enable MLflow autologging\n",
            "mlflow.sklearn.autolog()\n",
            "\n",
            "X = df.drop('Survived', axis=1)\n",
            "y = df['Survived']\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "\n",
            "with mlflow.start_run(run_name='Eksperimen_Dhafin_Lubis'):\n",
            "    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)\n",
            "    model.fit(X_train, y_train)\n",
            "    y_pred = model.predict(X_test)\n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    print(f'Accuracy: {acc}')\n"
        ]
    }
]

nb['cells'].extend(new_cells)

with open('Eksperimen_Dhafin-Lubis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
