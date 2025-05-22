import streamlit as st
import numpy as np
import pandas as pd
from model import MyLogisticRegression

# Ініціалізація моделі
model = MyLogisticRegression()

st.title("🚢 Titanic Survival Prediction")
st.markdown("Введи характеристики пасажира:")

# Ввід ознак
Pclass = st.selectbox("Клас", [1, 2, 3])
Sex = st.selectbox("Стать", ["чоловік", "жінка"])
Age = st.slider("Вік", 0, 80, 25)
SibSp = st.number_input("К-ть братів/сестер або чоловіка/дружини", 0, 10, 0)
Parch = st.number_input("К-ть батьків/дітей", 0, 10, 0)
Fare = st.slider("Ціна квитка", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Порт посадки", ["S", "C", "Q"])

# Обчислення фічей
FamilySize = SibSp + Parch + 1
IsAlone = int(FamilySize == 1)

# Перетворення вхідних даних у формат моделі
features = {
    'Age': Age,
    'SibSp': SibSp,
    'Parch': Parch,
    'Fare': Fare,
    'FamilySize': FamilySize,
    'IsAlone': IsAlone,
    'male': 1 if Sex == "чоловік" else 0,
    'Q': 1 if Embarked == "Q" else 0,
    'S': 1 if Embarked == "S" else 0,
    'Pclass_2': 1 if Pclass == 2 else 0,
    'Pclass_3': 1 if Pclass == 3 else 0
}

X_input = pd.DataFrame([features])
X_array = X_input.to_numpy()

# Передбачення
if st.button("🔮 Передбачити"):
    result = model.predict(X_array)[0]
    prob = model.predict_prob(X_array)[0]
    msg = "✅ Вижив" if result == 1 else "❌ Не вижив"
    st.subheader(f"Результат: {msg}")
    st.markdown(f"**Ймовірність виживання:** {prob:.2%}")
