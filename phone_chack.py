#!/usr/bin/env python3
import sys
import json
import requests
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit,
                             QComboBox, QCheckBox, QGroupBox)
from PyQt6.QtCore import Qt


class VoxLinkTester(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('VoxLink Num API Tester')
        self.setGeometry(300, 300, 600, 500)

        # Основной вертикальный лайаут
        main_layout = QVBoxLayout()

        # --- Блок настроек запроса ---
        settings_group = QGroupBox("Параметры запроса")
        settings_layout = QVBoxLayout()

        # Поле ввода номера
        input_layout = QHBoxLayout()
        self.num_input = QLineEdit()
        self.num_input.setPlaceholderText("Например: 79000000000 или +7...")
        self.num_input.setText("74993809706")  # Дефолт из примера
        input_layout.addWidget(QLabel("Номер (num):"))
        input_layout.addWidget(self.num_input)
        settings_layout.addLayout(input_layout)

        # Выбор поля (field) и транслит
        params_layout = QHBoxLayout()

        self.field_combo = QComboBox()
        # Первый пункт - пустой (для получения полного JSON)
        self.field_combo.addItem("Все данные (JSON)", None)
        # Остальные поля API
        fields = ["operator", "region", "code", "num", "full_num", "old_operator"]
        for f in fields:
            self.field_combo.addItem(f"Поле: {f}", f)

        self.translit_check = QCheckBox("Транслит (translit=1)")
        self.translit_check.setEnabled(False)  # Активен только если выбрано поле field

        # Логика UI: транслит работает только с параметром field
        self.field_combo.currentIndexChanged.connect(self.toggle_translit)

        params_layout.addWidget(QLabel("Формат ответа:"))
        params_layout.addWidget(self.field_combo)
        params_layout.addWidget(self.translit_check)
        settings_layout.addLayout(params_layout)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # --- Кнопка отправки ---
        self.btn_send = QPushButton("Отправить запрос")
        self.btn_send.setStyleSheet("background-color: #2e86de; color: white; font-weight: bold; padding: 8px;")
        self.btn_send.clicked.connect(self.make_request)
        main_layout.addWidget(self.btn_send)

        # --- Блок вывода ---
        # Поле для отображения сформированного URL (для отладки)
        self.url_display = QLineEdit()
        self.url_display.setReadOnly(True)
        self.url_display.setStyleSheet("color: gray; font-family: monospace;")
        main_layout.addWidget(QLabel("Сформированный URL:"))
        main_layout.addWidget(self.url_display)

        # Поле для ответа
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("font-family: monospace; font-size: 12px;")
        main_layout.addWidget(QLabel("Ответ сервера:"))
        main_layout.addWidget(self.result_area)

        self.setLayout(main_layout)

    def toggle_translit(self):
        """Включает чекбокс транслита только если выбрано конкретное поле (field)."""
        field_value = self.field_combo.currentData()
        self.translit_check.setEnabled(field_value is not None)
        if field_value is None:
            self.translit_check.setChecked(False)

    def make_request(self):
        """Сборка параметров и выполнение запроса."""
        base_url = "http://num.voxlink.ru/get/"
        phone_num = self.num_input.text().strip()

        if not phone_num:
            self.result_area.setText("Ошибка: Введите номер телефона.")
            return

        params = {"num": phone_num}

        # Добавляем field если выбран
        field_val = self.field_combo.currentData()
        if field_val:
            params["field"] = field_val
            # Добавляем translit только если есть field (согласно документации)
            if self.translit_check.isChecked():
                params["translit"] = "1"

        try:
            # Формируем объект Request чтобы показать пользователю итоговый URL
            req = requests.Request('GET', base_url, params=params)
            prepped = req.prepare()
            self.url_display.setText(prepped.url)

            # Выполняем запрос
            # Примечание: для production-GUI лучше использовать QThread,
            # но для простой утилиты синхронный вызов допустим.
            response = requests.Session().send(prepped, timeout=5)
            response.raise_for_status()  # Проверка на ошибки HTTP (4xx, 5xx)

            # Обработка ответа
            content_type = response.headers.get('Content-Type', '')

            # Если вернулся JSON (режим "Все данные")
            if field_val is None:
                try:
                    data = response.json()
                    formatted_json = json.dumps(data, indent=4, ensure_ascii=False)

                    # Подсветка для MNP (если номер был перенесен)
                    if "old_operator" in data:
                        header = ">>> ОБНАРУЖЕН ПЕРЕНОС НОМЕРА (MNP) <<<\n"
                        self.result_area.setText(header + formatted_json)
                    else:
                        self.result_area.setText(formatted_json)
                except json.JSONDecodeError:
                    self.result_area.setText(f"Ошибка парсинга JSON:\n{response.text}")

            # Если вернулся Plain Text (режим field)
            else:
                self.result_area.setText(response.text)

        except requests.exceptions.RequestException as e:
            self.result_area.setText(f"Ошибка сети:\n{str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoxLinkTester()
    window.show()
    sys.exit(app.exec())
