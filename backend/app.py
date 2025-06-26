from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd
import catboost
# Inicializar la app
app = Flask(__name__)
CORS(app)

# Cargar modelos
modelo_aspirados = joblib.load("C:/Users/crf16/OneDrive/Escritorio/tfm/modelo_xgb_residuos.pkl")
modelo_mii=joblib.load("C:/Users/crf16/OneDrive/Escritorio/tfm/modelo_CatBoostClassifier (1).pkl")
modelo_calidad = joblib.load("C:/Users/crf16/OneDrive/Escritorio/tfm/model_calidad_grupo (2).pkl")

# Función para generar sugerencias de mejora
def sugerencias_mejora(X, modelo, feature_names, modificables):
    original_pred = modelo.predict([X])[0]
    sugerencias = {}

    for var in modificables:
        idx = feature_names.index(var)
        valor_original = X[idx]

        # Solo intentar si el valor es numérico
        if isinstance(valor_original, (int, float)):
            mejor_delta = None
            mejor_mejora = 0.0

            for delta in [-300, -100, -10, -5, -1, -0.5, 0.5, 1, 5, 10, 100, 300]:
                X_mod = X.copy()
                X_mod[idx] = valor_original + delta
                try:
                    nueva_pred = modelo.predict([X_mod])[0]
                    diferencia = float(nueva_pred) - float(original_pred)
                except:
                    continue

                if diferencia > mejor_mejora:
                    mejor_mejora = diferencia
                    mejor_delta = delta

            if mejor_mejora > 0.5:
                sugerencias[var] = {
                    'cambio_sugerido': mejor_delta,
                    'mejora_prediccion': round(float(mejor_mejora), 2)
                }

    return sugerencias






@app.route('/predecir/aspirados', methods=['POST'])
def predecir_aspirados():
    data = request.get_json()
    try:
        feature_names = [
            'edad', 'bmi', 'amh', 'n_folic_antral_der', 'n_folic_antral_izq',
            'dosis_total', 'rfa', 'estim_final_cod', 'dias_estim_e_cl',
            'p4_dhcg_de', 'nivel_fsh_h_de', 'ult_lin_endom',
            'amp_fsh_e_cl', 'amp_lh_e_cl'
        ]
        modificables = ['dosis_total', 'rfa', 'dias_estim_e_cl', 'amp_fsh_e_cl', 'amp_lh_e_cl','estim_final_cod']

        X = np.array([[
            data['edad'], data['bmi'], data['amh'],
            data['n_folic_antral_der'], data['n_folic_antral_izq'],
            data['dosis_total'], data['rfa'], data['estim_final_cod'],
            data['dias_estim_e_cl'], data['p4_dhcg_de'], data['nivel_fsh_h_de'],
            data['ult_lin_endom'], data['amp_fsh_e_cl'], data['amp_lh_e_cl']
        ]])

        pred = modelo_aspirados.predict(X)[0]
        sugerencias = sugerencias_mejora(X[0], modelo_aspirados, feature_names, modificables)
        return jsonify({'prediccion': round(float(pred), 2), 'porcentaje_acierto': 93, 'sugerencias_mejora': sugerencias})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predecir/calidad', methods=['POST'])
def predecir_calidad():
    data = request.get_json()
    print("Datos recibidos en CALIDAD:", data) 
    try:
        feature_names = [
        'edad', 'bmi', 'amh', 'n_folic_antral_der', 'n_folic_antral_izq',
        'dosis_total', 'rfa', 'estim_final_cod', 'aspirados_predict',
        'nivel_fsh_h_de', 'p4_dhcg_de', 'dias_estim_e_cl',
        'amp_fsh_e_cl', 'amp_lh_e_cl', 'd2_celulas.-', 'd3_celulas.-', 'dia5.-','Maduro_grupo'
    ]


        modificables = ['dosis_total', 'rfa', 'dias_estim_e_cl',
    'amp_fsh_e_cl', 'amp_lh_e_cl',
    'd2_celulas.-', 'd3_celulas.-', 'dia5.-']

        
        X = pd.DataFrame([[
    data['PATIENT AGE'], data['PATIENT BMI'], data['ANTIUMULLERIANA_DE'],
    data['N_FOLIC_ANTRAL_DER_IN.-'], data['N_FOLIC_ANTRAL_IZQ_IN.-'],
    data['DOSIS TOTAL'], data['RFA'], str(data['ESTIMULACIÓN FINAL COD']),
    data['ASPIRADOS_PREDICT'], data['NIVEL_FSH_H_DE.-_OK'],
    data['P4_DHCG_DE.-'], data['dias_estim_e_cl.-'],
    data['ampollas_fsh_e_cl.-'], data['ampollas_lh_e_cl.-'],
    str(data['d2_celulas.-']), str(data['d3_celulas.-']), str(data['dia5.-']),
    str(data['OVO_MII_INSEMIN_IN.-_OK_GRUPO'])
]], columns=[
    "PATIENT AGE", "PATIENT BMI", "ANTIUMULLERIANA_DE", "N_FOLIC_ANTRAL_DER_IN.-", 
    "N_FOLIC_ANTRAL_IZQ_IN.-", "DOSIS TOTAL", "RFA", "ESTIMULACIÓN FINAL COD",
    "ASPIRADOS_PREDICT", "NIVEL_FSH_H_DE.-_OK", "P4_DHCG_DE.-", "dias_estim_e_cl.-",
    "ampollas_fsh_e_cl.-", "ampollas_lh_e_cl.-", "d2_celulas.-", "d3_celulas.-", 
    "dia5.-", "OVO_MII_INSEMIN_IN.-_OK_GRUPO"
])

        pred = modelo_calidad.predict(X)
        pred_str = str(pred[0]) if isinstance(pred[0], str) else str(pred[0][0])
        print("Predicción real del modelo:", pred[0])

        if pred_str== "Maduros con alta calidad":
          sugerencias1 =   "Los ovocitos presentan buena calidad. Mantener el protocolo actual puede ser una buena estrategia. Evita realizar cambios bruscos a menos que clínicamente estén justificados."

        elif pred_str == "Maduros con media calidad":
               sugerencias1 = "La calidad ovocitaria observada es intermedia, lo que indica que, si bien se han obtenido ovocitos maduros, su morfología o características funcionales podrían no ser óptimas para una fecundación exitosa o un buen desarrollo embrionario. En estos casos, se recomienda revisar el protocolo de estimulación utilizado, ya que pequeños ajustes pueden marcar la diferen Por ejemplo, aumentar ligeramente la dosis de FSH podría mejorar la sincronización del crecimiento folicular, promoviendo una maduración más homogénea y eficaz de los ovocitos. También es clave evaluar el momento del disparo de hCG, ya que una administración demasiado temprana o tardía puede afectar negativamente la madurez nuclear y citoplasmática del ovocito. Un control más preciso de estos factores puede contribuir a mejorar la calidad final de los ovocitos obtenidos y, con ello, las probabilidades de éxito del tratamiento."

        elif pred_str == "Maduros con baja calidad":
            sugerencias1 = "La calidad ovocitaria es baja. Puede valorarse un cambio en el protocolo de estimulación, ajustar días de estimulación o considerar un aumento en las gonadotropinas."
        else:
            sugerencias1 = "No se ha podido interpretar correctamente la predicción. Revisa los datos introducidos."
        rangos_CALIDAD = {
            "Maduros con alta calidad": "[A-B]",
            "Maduros con media calidad": "[C-D]",
            "Maduros con baja calidad": "[E]"
        }

        rango = rangos_CALIDAD.get(pred_str, "")
        pred_completa = f"{pred_str} {rango}"

        return jsonify({
            'prediccion': pred_completa,
            'porcentaje_acierto': 80,
            'recomendacion_clinica': sugerencias1
        })

       
    except KeyError as e:
        print("❌ KeyError:", e)
        return jsonify({"error": f"Falta el campo {e} en los datos recibidos"}), 400
    except Exception as e:
        print("❌ Error general:", e)
        return jsonify({'error': str(e)}), 400
@app.route('/predecir/maduros', methods=['POST'])
def predecir_maduros():
    data = request.get_json()
    print("Datos recibidos en MADUROS:", data) 
    try:
        feature_names = [
        'edad', 'bmi', 'amh', 'n_folic_antral_der', 'n_folic_antral_izq',
        'dosis_total', 'rfa', 'estim_final_cod', 'aspirados_predict',
        'nivel_fsh_h_de', 'p4_dhcg_de', 'dias_estim_e_cl',
        'amp_fsh_e_cl', 'amp_lh_e_cl'
    ]


        modificables = ['dosis_total', 'rfa', 'dias_estim_e_cl',
    'amp_fsh_e_cl', 'amp_lh_e_cl']

        
        X = pd.DataFrame([[
    data['PATIENT AGE'], data['PATIENT BMI'], data['ANTIUMULLERIANA_DE'],
    data['N_FOLIC_ANTRAL_DER_IN.-'], data['N_FOLIC_ANTRAL_IZQ_IN.-'],
    data['DOSIS TOTAL'], data['RFA'], str(data['ESTIMULACIÓN FINAL COD']),
    data['ASPIRADOS_PREDICT'], data['NIVEL_FSH_H_DE.-_OK'],
    data['P4_DHCG_DE.-'], data['dias_estim_e_cl.-'],
    data['ampollas_fsh_e_cl.-'], data['ampollas_lh_e_cl.-']
]], columns=[
    "PATIENT AGE", "PATIENT BMI", "ANTIUMULLERIANA_DE", "N_FOLIC_ANTRAL_DER_IN.-", 
    "N_FOLIC_ANTRAL_IZQ_IN.-", "DOSIS TOTAL", "RFA", "ESTIMULACIÓN FINAL COD",
    "ASPIRADOS_PREDICT", "NIVEL_FSH_H_DE.-_OK", "P4_DHCG_DE.-", "dias_estim_e_cl.-",
    "ampollas_fsh_e_cl.-", "ampollas_lh_e_cl.-"
])

        pred = modelo_mii.predict(X)
        pred_str = str(pred[0]) if isinstance(pred[0], str) else str(pred[0][0])

        print("Predicción real del modelo:", pred[0])

        if pred_str == "Alto numero de maduracion":
            sugerencias1 = "Se ha obtenido un alto número de ovocitos maduros, lo cual refleja una respuesta ovárica favorable y una correcta sincronización del estímulo hormonal. En este contexto, no se requieren cambios significativos en el protocolo de estimulación. Es recomendable mantener la estrategia actual, asegurando siempre un seguimiento clínico individualizado para consolidar estos buenos resultados en ciclos futuros."

        elif pred_str == "Medio numero de maduracion":
            sugerencias1 = "El número de ovocitos maduros es intermedio, lo que sugiere una respuesta parcial al protocolo de estimulación. Esto podría deberse a una subestimulación leve o a una ligera desincronización del disparo de hCG. Se recomienda considerar ajustes como un aumento progresivo de la dosis de FSH o una revisión precisa del momento de administración de la hCG. Optimizar estos parámetros puede favorecer una maduración más completa y homogénea de los ovocitos, aumentando las tasas de éxito en procedimientos posteriores."

        elif pred_str == "Bajo numero de maduracion":
            sugerencias1 = "Se ha observado un bajo número de ovocitos maduros, lo que indica una maduración incompleta posiblemente asociada a una estimulación insuficiente o una planificación subóptima del ciclo. En estos casos, se aconseja reevaluar el protocolo hormonal, incrementando la dosis de gonadotropinas o ajustando la duración de la estimulación. También es fundamental revisar la precisión en el momento del disparo de hCG, ya que su administración fuera del intervalo óptimo puede comprometer seriamente la maduración ovocitaria."

        else:
            sugerencias1 = "No se ha podido interpretar correctamente la predicción. Revisa los datos introducidos o consulta con el equipo médico."

        rangos_maduracion = {
            "Bajo numero de maduracion": "[0-3]",
            "Medio numero de maduracion": "[4-7]",
            "Alto numero de maduracion": "[8-10]"
        }

        rango = rangos_maduracion.get(pred_str, "")
        pred_completa = f"{pred_str} {rango}"

        return jsonify({
            'prediccion': pred_completa,
            'porcentaje_acierto': 80,
            'recomendacion_clinica': sugerencias1
        })





    except KeyError as e:
        print("❌ KeyError:", e)
        return jsonify({"error": f"Falta el campo {e} en los datos recibidos"}), 400
    except Exception as e:
        print("❌ Error general:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
