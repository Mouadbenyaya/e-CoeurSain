import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

class HeartDiseasePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédiction de Maladie Cardiaque / التنبؤ بأمراض القلب")
        self.root.geometry("850x750")
        self.root.configure(bg="#f0f0f0")
        
        self.model_path = "heart_disease_xgboost_model.json"
        if not os.path.exists(self.model_path):
            messagebox.showerror("Erreur / خطأ", f"Le modèle '{self.model_path}' est introuvable. / النموذج غير موجود")
            self.model_loaded = False
        else:
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                self.model_loaded = True
            except Exception as e:
                messagebox.showerror("Erreur / خطأ", f"Impossible de charger le modèle: {str(e)} / لا يمكن تحميل النموذج")
                self.model_loaded = False
        
        self.scaler = StandardScaler()
        
        self.feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Dictionnaire de traductions
        self.translations = {
            'fr': {
                'title': "Prédiction de Maladie Cardiaque",
                'age': "Âge:",
                'sex': "Sexe:",
                'sex_values': ["Femme (0)", "Homme (1)"],
                'cp': "Type de douleur thoracique:",
                'cp_values': ["Angine typique (0)", "Angine atypique (1)", 
                              "Douleur non angineuse (2)", "Asymptomatique (3)"],
                'trestbps': "Pression artérielle au repos (mmHg):",
                'chol': "Cholestérol sérique (mg/dl):",
                'fbs': "Sucre sanguin à jeun > 120 mg/dl:",
                'fbs_values': ["Non (0)", "Oui (1)"],
                'restecg': "Résultat ECG au repos:",
                'restecg_values': ["Normal (0)", "Anomalie ST-T (1)", 
                                  "Hypertrophie ventriculaire (2)"],
                'thalach': "Fréquence cardiaque maximale:",
                'exang': "Angine induite par l'exercice:",
                'exang_values': ["Non (0)", "Oui (1)"],
                'oldpeak': "Oldpeak (dépression ST):",
                'slope': "Pente segment ST:",
                'slope_values': ["Ascendante (0)", "Plate (1)", "Descendante (2)"],
                'ca': "Nombre de vaisseaux colorés (0-3):",
                'ca_values': ["0", "1", "2", "3"],
                'thal': "Thal:",
                'thal_values': ["Normal (1)", "Défaut fixe (2)", "Défaut réversible (3)"],
                'predict': "Prédire",
                'result': "Résultat:",
                'probability': "Probabilité:",
                'positive': "Maladie cardiaque détectée",
                'negative': "Pas de maladie cardiaque",
                'language': "Langue:",
                'language_values': ["Français", "العربية"]
            },
            'ar': {
                'title': "التنبؤ بأمراض القلب",
                'age': "العمر:",
                'sex': "الجنس:",
                'sex_values': ["أنثى (0)", "ذكر (1)"],
                'cp': "نوع ألم الصدر:",
                'cp_values': ["ذبحة صدرية نموذجية (0)", "ذبحة صدرية غير نموذجية (1)", 
                              "ألم غير ذبحة صدرية (2)", "بدون أعراض (3)"],
                'trestbps': "ضغط الدم في وضع الراحة (ملم زئبق):",
                'chol': "الكوليسترول في الدم (ملغ/ديسيلتر):",
                'fbs': "سكر الدم الصيامي > 120 ملغ/ديسيلتر:",
                'fbs_values': ["لا (0)", "نعم (1)"],
                'restecg': "نتيجة تخطيط القلب الكهربائي في الراحة:",
                'restecg_values': ["طبيعي (0)", "شذوذ في قطعة ST-T (1)", 
                                   "تضخم البطين (2)"],
                'thalach': "أقصى معدل لضربات القلب:",
                'exang': "ذبحة صدرية ناتجة عن التمرين:",
                'exang_values': ["لا (0)", "نعم (1)"],
                'oldpeak': "انخفاض قطعة ST:",
                'slope': "منحدر قطعة ST:",
                'slope_values': ["صاعد (0)", "مستوٍ (1)", "نازل (2)"],
                'ca': "عدد الأوعية الدموية الملونة (0-3):",
                'ca_values': ["0", "1", "2", "3"],
                'thal': "فحص الثاليوم:",
                'thal_values': ["طبيعي (1)", "عيب ثابت (2)", "عيب قابل للانعكاس (3)"],
                'predict': "تنبؤ",
                'result': "النتيجة:",
                'probability': "الاحتمالية:",
                'positive': "تم اكتشاف مرض القلب",
                'negative': "لا يوجد مرض قلبي",
                'language': "اللغة:",
                'language_values': ["Français", "العربية"]
            }
        }
        
        # Langue par défaut
        self.current_language = 'fr'
        
        self.create_ui()
    
    def create_ui(self):
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 11), background="#f0f0f0")
        style.configure('TButton', font=('Arial', 12, 'bold'))
        style.configure('TCombobox', font=('Arial', 11))
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), background="#f0f0f0")
        style.configure('Result.TLabel', font=('Arial', 12), background="#f0f0f0")
        style.configure('ResultValue.TLabel', font=('Arial', 16, 'bold'), background="#f0f0f0")
        
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        lang_frame = ttk.Frame(self.main_frame)
        lang_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        ttk.Label(lang_frame, text=self.translations[self.current_language]['language'], style='TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.language_selector = ttk.Combobox(lang_frame, width=10, state="readonly", 
                                          values=self.translations[self.current_language]['language_values'])
        self.language_selector.pack(side=tk.LEFT)
        self.language_selector.current(0)
        self.language_selector.bind("<<ComboboxSelected>>", self.change_language)
        
        self.title_label = ttk.Label(self.main_frame, text=self.translations[self.current_language]['title'], style='Header.TLabel')
        self.title_label.grid(row=1, column=0, columnspan=4, pady=(0, 20))
        
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        self.labels = {}
        self.entries = {}
        
        self.create_input_widgets()
        
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=3, column=0, pady=20)
        
        self.predict_button = ttk.Button(self.button_frame, text=self.translations[self.current_language]['predict'], 
                                   command=self.predict, style='TButton')
        self.predict_button.pack(pady=10)
        
        self.result_frame = ttk.Frame(self.main_frame, padding="20")
        self.result_frame.grid(row=4, column=0, sticky="nsew", pady=10)
        
        self.result_text_label = ttk.Label(self.result_frame, text=self.translations[self.current_language]['result'], style='Result.TLabel')
        self.result_text_label.grid(row=0, column=0, sticky="w")
        
        self.result_label = ttk.Label(self.result_frame, text="", style='ResultValue.TLabel')
        self.result_label.grid(row=0, column=1, sticky="w", padx=10)
        
        self.probability_text_label = ttk.Label(self.result_frame, text=self.translations[self.current_language]['probability'], style='Result.TLabel')
        self.probability_text_label.grid(row=1, column=0, sticky="w", pady=10)
        
        self.probability_label = ttk.Label(self.result_frame, text="", style='ResultValue.TLabel')
        self.probability_label.grid(row=1, column=1, sticky="w", padx=10, pady=10)
        
        self.set_default_values()
    
    def create_input_widgets(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
            
        lang = self.current_language
        
        self.labels['age'] = ttk.Label(self.input_frame, text=self.translations[lang]['age'])
        self.labels['age'].grid(row=0, column=0, sticky="w", pady=5)
        self.entries['age'] = ttk.Entry(self.input_frame, width=15)
        self.entries['age'].grid(row=0, column=1, sticky="w", pady=5)
        
        self.labels['sex'] = ttk.Label(self.input_frame, text=self.translations[lang]['sex'])
        self.labels['sex'].grid(row=1, column=0, sticky="w", pady=5)
        self.entries['sex'] = ttk.Combobox(self.input_frame, width=20, state="readonly", 
                                          values=self.translations[lang]['sex_values'])
        self.entries['sex'].grid(row=1, column=1, sticky="w", pady=5)
        self.entries['sex'].current(0)
        
        # Type de douleur thoracique
        self.labels['cp'] = ttk.Label(self.input_frame, text=self.translations[lang]['cp'])
        self.labels['cp'].grid(row=2, column=0, sticky="w", pady=5)
        self.entries['cp'] = ttk.Combobox(self.input_frame, width=30, state="readonly", 
                                         values=self.translations[lang]['cp_values'])
        self.entries['cp'].grid(row=2, column=1, sticky="w", pady=5)
        self.entries['cp'].current(0)
        
        self.labels['trestbps'] = ttk.Label(self.input_frame, text=self.translations[lang]['trestbps'])
        self.labels['trestbps'].grid(row=3, column=0, sticky="w", pady=5)
        self.entries['trestbps'] = ttk.Entry(self.input_frame, width=15)
        self.entries['trestbps'].grid(row=3, column=1, sticky="w", pady=5)
        
        self.labels['chol'] = ttk.Label(self.input_frame, text=self.translations[lang]['chol'])
        self.labels['chol'].grid(row=4, column=0, sticky="w", pady=5)
        self.entries['chol'] = ttk.Entry(self.input_frame, width=15)
        self.entries['chol'].grid(row=4, column=1, sticky="w", pady=5)
        
       
        self.labels['fbs'] = ttk.Label(self.input_frame, text=self.translations[lang]['fbs'])
        self.labels['fbs'].grid(row=0, column=2, sticky="w", pady=5, padx=(20, 0))
        self.entries['fbs'] = ttk.Combobox(self.input_frame, width=20, state="readonly", 
                                         values=self.translations[lang]['fbs_values'])
        self.entries['fbs'].grid(row=0, column=3, sticky="w", pady=5)
        self.entries['fbs'].current(0)
        
        self.labels['restecg'] = ttk.Label(self.input_frame, text=self.translations[lang]['restecg'])
        self.labels['restecg'].grid(row=1, column=2, sticky="w", pady=5, padx=(20, 0))
        self.entries['restecg'] = ttk.Combobox(self.input_frame, width=30, state="readonly", 
                                              values=self.translations[lang]['restecg_values'])
        self.entries['restecg'].grid(row=1, column=3, sticky="w", pady=5)
        self.entries['restecg'].current(0)
        
        self.labels['thalach'] = ttk.Label(self.input_frame, text=self.translations[lang]['thalach'])
        self.labels['thalach'].grid(row=2, column=2, sticky="w", pady=5, padx=(20, 0))
        self.entries['thalach'] = ttk.Entry(self.input_frame, width=15)
        self.entries['thalach'].grid(row=2, column=3, sticky="w", pady=5)
        
        self.labels['exang'] = ttk.Label(self.input_frame, text=self.translations[lang]['exang'])
        self.labels['exang'].grid(row=3, column=2, sticky="w", pady=5, padx=(20, 0))
        self.entries['exang'] = ttk.Combobox(self.input_frame, width=20, state="readonly", 
                                           values=self.translations[lang]['exang_values'])
        self.entries['exang'].grid(row=3, column=3, sticky="w", pady=5)
        self.entries['exang'].current(0)
        
        self.labels['oldpeak'] = ttk.Label(self.input_frame, text=self.translations[lang]['oldpeak'])
        self.labels['oldpeak'].grid(row=4, column=2, sticky="w", pady=5, padx=(20, 0))
        self.entries['oldpeak'] = ttk.Entry(self.input_frame, width=15)
        self.entries['oldpeak'].grid(row=4, column=3, sticky="w", pady=5)
        
        self.labels['slope'] = ttk.Label(self.input_frame, text=self.translations[lang]['slope'])
        self.labels['slope'].grid(row=5, column=0, sticky="w", pady=5)
        self.entries['slope'] = ttk.Combobox(self.input_frame, width=30, state="readonly", 
                                           values=self.translations[lang]['slope_values'])
        self.entries['slope'].grid(row=5, column=1, sticky="w", pady=5)
        self.entries['slope'].current(0)
        
        self.labels['ca'] = ttk.Label(self.input_frame, text=self.translations[lang]['ca'])
        self.labels['ca'].grid(row=5, column=2, sticky="w", pady=5, padx=(20, 0))
        self.entries['ca'] = ttk.Combobox(self.input_frame, width=15, state="readonly", 
                                        values=self.translations[lang]['ca_values'])
        self.entries['ca'].grid(row=5, column=3, sticky="w", pady=5)
        self.entries['ca'].current(0)
        
        self.labels['thal'] = ttk.Label(self.input_frame, text=self.translations[lang]['thal'])
        self.labels['thal'].grid(row=6, column=0, sticky="w", pady=5)
        self.entries['thal'] = ttk.Combobox(self.input_frame, width=30, state="readonly", 
                                          values=self.translations[lang]['thal_values'])
        self.entries['thal'].grid(row=6, column=1, sticky="w", pady=5)
        self.entries['thal'].current(0)
    
    def change_language(self, event=None):
        selected_index = self.language_selector.current()
        if selected_index == 0:
            self.current_language = 'fr'
        else:
            self.current_language = 'ar'
        
        self.update_ui_language()
    
    def update_ui_language(self):
        lang = self.current_language
        
        self.title_label.config(text=self.translations[lang]['title'])
        
        self.predict_button.config(text=self.translations[lang]['predict'])
        
        self.result_text_label.config(text=self.translations[lang]['result'])
        self.probability_text_label.config(text=self.translations[lang]['probability'])
        
        self.create_input_widgets()
        
        self.set_default_values()
    
    def set_default_values(self):
        """Définir des valeurs par défaut pour les champs"""
        default_values = {
            'age': '60',
            'trestbps': '130',
            'chol': '250',
            'thalach': '150',
            'oldpeak': '1.5'
        }
        
        for key, value in default_values.items():
            if key in self.entries and not self.entries[key].get():
                self.entries[key].insert(0, value)
                
    def get_input_values(self):
        """Extraire et valider les valeurs des champs de saisie"""
        try:
            data = {}
            
            # Âge
            data['age'] = float(self.entries['age'].get())
            
            # Sexe (extraire le chiffre entre parenthèses)
            sex_value = self.entries['sex'].get()
            data['sex'] = int(sex_value[-2:-1])
            
            # Type de douleur thoracique
            cp_value = self.entries['cp'].get()
            data['cp'] = int(cp_value[-2:-1])
            
            # Pression artérielle
            data['trestbps'] = float(self.entries['trestbps'].get())
            
            # Cholestérol
            data['chol'] = float(self.entries['chol'].get())
            
            # Glycémie à jeun
            fbs_value = self.entries['fbs'].get()
            data['fbs'] = int(fbs_value[-2:-1])
            
            # ECG au repos
            restecg_value = self.entries['restecg'].get()
            data['restecg'] = int(restecg_value[-2:-1])
            
            # Fréquence cardiaque max
            data['thalach'] = float(self.entries['thalach'].get())
            
            # Angine d'effort
            exang_value = self.entries['exang'].get()
            data['exang'] = int(exang_value[-2:-1])
            
            # Oldpeak
            data['oldpeak'] = float(self.entries['oldpeak'].get())
            
            # Pente ST
            slope_value = self.entries['slope'].get()
            data['slope'] = int(slope_value[-2:-1])
            
            # Nombre de vaisseaux colorés
            data['ca'] = int(self.entries['ca'].get())
            
            # Thal
            thal_value = self.entries['thal'].get()
            data['thal'] = int(thal_value[-2:-1])
            
            return data
            
        except ValueError as e:
            error_message = "Veuillez vérifier vos entrées numériques" if self.current_language == 'fr' else "يرجى التحقق من المدخلات الرقمية"
            messagebox.showerror("Erreur / خطأ", f"{error_message}: {str(e)}")
            return None
        except Exception as e:
            error_message = "Une erreur s'est produite" if self.current_language == 'fr' else "حدث خطأ"
            messagebox.showerror("Erreur / خطأ", f"{error_message}: {str(e)}")
            return None
    
    def predict(self):
        """Effectuer la prédiction basée sur les entrées de l'utilisateur"""
        if not self.model_loaded:
            error_message = "Le modèle n'est pas chargé" if self.current_language == 'fr' else "لم يتم تحميل النموذج"
            messagebox.showerror("Erreur / خطأ", error_message)
            return
            
        # Obtenir les valeurs d'entrée
        input_data = self.get_input_values()
        if input_data is None:
            return
            
        try:
            df = pd.DataFrame([input_data])
            
            df = df[self.feature_columns]
            
            # Faire la prédiction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]  # Probabilité de la classe positive (1)
            
            # Afficher les résultats
            if prediction == 1:
                self.result_label.config(text=self.translations[self.current_language]['positive'], foreground="red")
            else:
                self.result_label.config(text=self.translations[self.current_language]['negative'], foreground="green")
                
            self.probability_label.config(text=f"{probability*100:.2f}%")
            
        except Exception as e:
            error_message = "Impossible de faire une prédiction" if self.current_language == 'fr' else "لا يمكن إجراء التنبؤ"
            messagebox.showerror("Erreur / خطأ", f"{error_message}: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseasePredictor(root)
    root.mainloop()