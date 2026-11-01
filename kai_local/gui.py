"""Local desktop UI. No web server, external calls, authentication or publication."""
import json
import math
import os
import queue
import re
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import gui_jobs

ROOT = Path(__file__).resolve().parent.parent
TITLES = {'vocales': 'Vocales', 'abecedario': 'Abecedario', 'numeros': 'Números', 'matematicas': 'Matemáticas'}
BG, CARD, INK, MUTED, PURPLE = '#f4f4f9', '#ffffff', '#29283d', '#666579', '#4a47a3'


class TrainerWindow:
    def __init__(self, root):
        self.root = root
        self.paths = []
        self.audit = None
        self.busy = False
        self.closing = False
        self.cancel_file = None
        self.result_dir = None
        self.events = queue.Queue()
        self.editable = []
        root.title('Kai Señas | Entrenamiento local')
        root.geometry('1160x810')
        root.minsize(960, 700)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background=BG, foreground=INK)
        style.configure('TButton', padding=(12, 9), background=CARD, borderwidth=1)
        style.configure('Primary.TButton', background=PURPLE, foreground='white', padding=(12, 10))
        style.map('Primary.TButton', background=[('disabled', '#a5a3cf'), ('active', '#38357e')])
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(18, 10), background='#e8e7f2')
        style.map('TNotebook.Tab', background=[('selected', CARD)])
        style.configure('Treeview', rowheight=29, background=CARD, fieldbackground=CARD, borderwidth=0)
        style.configure('Treeview.Heading', background='#eae9f5', font=('Segoe UI', 9, 'bold'), padding=8)
        style.configure('Horizontal.TProgressbar', background=PURPLE, troughcolor='#e4e3ed')
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        sidebar = tk.Frame(root, bg=PURPLE, padx=22, pady=26, width=255)
        sidebar.grid(row=0, column=0, sticky='ns')
        sidebar.grid_propagate(False)
        def side(text, size=10, bold=False):
            label = tk.Label(sidebar, text=text, bg=PURPLE, fg='white', font=('Segoe UI', size, 'bold' if bold else 'normal'), anchor='w', justify='left', wraplength=210)
            label.pack(fill='x', pady=(0, 9))
            return label
        side('Kai.', 32, True)
        side('LABORATORIO LOCAL', 9, True)
        side('Captura en la web.\nEntrena aquí.\nPublica con aprobación.', 11)
        side('Módulo', bold=True)
        self.category = tk.StringVar(value='vocales')
        self.category_box = ttk.Combobox(sidebar, textvariable=self.category, values=list(TITLES), state='readonly', width=20)
        self.category_box.pack(fill='x', pady=(0, 16))
        self.category_box.bind('<<ComboboxSelected>>', self.change_category)
        side('Nombre de versión', bold=True)
        self.version = tk.StringVar(value=self.new_version())
        self.version_entry = ttk.Entry(sidebar, textvariable=self.version)
        self.version_entry.pack(fill='x', pady=(0, 16))
        self.editable.append(self.version_entry)
        side('Etiquetas (opcional)', bold=True)
        self.labels = tk.StringVar()
        labels_entry = ttk.Entry(sidebar, textvariable=self.labels)
        labels_entry.pack(fill='x', pady=(0, 6))
        self.editable.append(labels_entry)
        side('Vacío: módulo completo.\nParcial: B, C, Nulo', 9)
        side('Configuración', bold=True)
        self.epochs, self.batch, self.threads = tk.StringVar(value='50'), tk.StringVar(value='16'), tk.StringVar(value='2')
        for title, variable, maximum in [('Épocas', self.epochs, 1000), ('Lote', self.batch, 1024), ('Hilos CPU', self.threads, 16)]:
            row = tk.Frame(sidebar, bg=PURPLE)
            row.pack(fill='x', pady=4)
            tk.Label(row, text=title, bg=PURPLE, fg='white', font=('Segoe UI', 9)).pack(side='left')
            spin = ttk.Spinbox(row, from_=1, to=maximum, width=6, textvariable=variable)
            spin.pack(side='right')
            self.editable.append(spin)
        tk.Label(sidebar, text='El entrenamiento no publica\nni modifica las prácticas.', bg=PURPLE, fg='#e2e0ff', font=('Segoe UI', 9), justify='left').pack(side='bottom', anchor='w', pady=12)
        main = tk.Frame(root, bg=BG, padx=25, pady=22)
        main.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)
        tk.Label(main, text='TU ESPACIO DE ENTRENAMIENTO', bg=BG, fg=PURPLE, font=('Segoe UI', 9, 'bold'), anchor='w').grid(row=0, column=0, sticky='ew')
        tk.Label(main, text='De tus capturas a un nuevo modelo.', bg=BG, fg=INK, font=('Segoe UI', 22, 'bold'), anchor='w').grid(row=1, column=0, sticky='ew', pady=(5, 16))
        bar = tk.Frame(main, bg=BG)
        bar.grid(row=2, column=0, sticky='ew')
        self.audit_button = ttk.Button(bar, text='1. Auditar capturas', command=self.start_audit)
        self.audit_button.pack(side='left', padx=(0, 8))
        self.train_button = ttk.Button(bar, text='2. Entrenar modelo', style='Primary.TButton', command=self.start_training, state='disabled')
        self.train_button.pack(side='left', padx=(0, 8))
        self.cancel_button = ttk.Button(bar, text='Cancelar', command=self.cancel, state='disabled')
        self.cancel_button.pack(side='left')
        self.status = tk.StringVar(value='Selecciona tus archivos y comienza por la auditoría.')
        tk.Label(main, textvariable=self.status, bg=BG, fg=MUTED, font=('Segoe UI', 10), wraplength=780, anchor='w', justify='left').grid(row=3, column=0, sticky='ew', pady=(12, 14))
        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=4, column=0, sticky='nsew')
        self.files_tab = tk.Frame(self.tabs, bg=CARD, padx=18, pady=18)
        self.training_tab = tk.Frame(self.tabs, bg=CARD, padx=18, pady=18)
        self.results_tab = tk.Frame(self.tabs, bg=CARD, padx=18, pady=18)
        for tab, text in [(self.files_tab, 'Capturas'), (self.training_tab, 'Entrenamiento'), (self.results_tab, 'Resultados')]:
            self.tabs.add(tab, text=text)
        self.build_files()
        self.build_training()
        self.build_results()
        self.labels.trace_add('write', lambda *args: self.invalidate())
        self.root.after(100, self.poll)
        self.load_default()

    def new_version(self):
        return self.category.get() + '-' + datetime.now().strftime('%Y%m%d-%H%M%S')

    def build_files(self):
        tab = self.files_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        buttons = tk.Frame(tab, bg=CARD)
        buttons.grid(row=0, column=0, sticky='ew', pady=(0, 12))
        for text, callback in [('Añadir archivos', self.add_files), ('Quitar seleccionados', self.remove_files), ('Vaciar lista', self.clear_files)]:
            button = ttk.Button(buttons, text=text, command=callback)
            button.pack(side='left', padx=(0, 7))
            self.editable.append(button)
        self.files_tree = ttk.Treeview(tab, columns=('file', 'size'), show='headings', height=6)
        self.files_tree.heading('file', text='Archivo de captura')
        self.files_tree.heading('size', text='Tamaño')
        self.files_tree.column('file', width=500)
        self.files_tree.column('size', width=90, stretch=False)
        self.files_tree.grid(row=1, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(tab, orient='vertical', command=self.files_tree.yview)
        scroll.grid(row=1, column=1, sticky='ns')
        self.files_tree.configure(yscrollcommand=scroll.set)
        self.summary = tk.StringVar(value='Sin archivos seleccionados.')
        tk.Label(tab, textvariable=self.summary, bg=CARD, fg=PURPLE, font=('Segoe UI', 11, 'bold'), anchor='w', wraplength=750).grid(row=2, column=0, sticky='ew', pady=(14, 6))
        self.audit_text = tk.Text(tab, height=9, wrap='word', bg='#f4f4f9', fg=INK, relief='flat', padx=12, pady=10, font=('Segoe UI', 10), state='disabled')
        self.audit_text.grid(row=3, column=0, sticky='ew')
        self.set_text(self.audit_text, 'Necesitas capturas independientes con todas las etiquetas elegidas, incluyendo Nulo. Copiar un archivo no agrega otra sesión.\n\nLos archivos se separan entre entrenamiento, validación y test. La auditoría valida los datos, no la precisión del futuro modelo.')

    def build_training(self):
        tab = self.training_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.epoch_text = tk.StringVar(value='Todavía no hay un entrenamiento en curso.')
        tk.Label(tab, textvariable=self.epoch_text, bg=CARD, fg=INK, font=('Segoe UI', 12, 'bold'), anchor='w').grid(row=0, column=0, sticky='ew')
        self.progress = ttk.Progressbar(tab, mode='determinate')
        self.progress.grid(row=1, column=0, sticky='ew', pady=14)
        self.log = tk.Text(tab, wrap='word', bg='#24233a', fg='#ecebf8', relief='flat', padx=12, pady=12, font=('Consolas', 9), state='disabled')
        self.log.grid(row=2, column=0, sticky='nsew')
        log_scroll = ttk.Scrollbar(tab, orient='vertical', command=self.log.yview)
        log_scroll.grid(row=2, column=1, sticky='ns')
        self.log.configure(yscrollcommand=log_scroll.set)
        tk.Label(tab, text='Cancelar espera un punto seguro; no fuerza el cierre de TensorFlow. El registro completo se guarda en artifacts/gui-logs.', bg=CARD, fg=MUTED, wraplength=740, justify='left', font=('Segoe UI', 9)).grid(row=3, column=0, sticky='ew', pady=(12, 0))

    def build_results(self):
        tab = self.results_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)
        ttk.Button(tab, text='Abrir informe anterior', command=self.open_report).grid(row=0, column=0, sticky='w', pady=(0, 12))
        self.result_summary = tk.StringVar(value='Los resultados aparecerán al terminar el entrenamiento.')
        tk.Label(tab, textvariable=self.result_summary, bg=CARD, fg=PURPLE, font=('Segoe UI', 13, 'bold'), anchor='w', wraplength=750, justify='left').grid(row=1, column=0, sticky='ew', pady=(0, 10))
        tk.Label(tab, text='Matriz de confusión: filas = etiqueta real; columnas = predicción. Revisa cada seña y Nulo antes de subir el paquete.', bg=CARD, fg=MUTED, wraplength=740, justify='left', font=('Segoe UI', 9)).grid(row=2, column=0, sticky='ew', pady=(0, 10))
        self.matrix = ttk.Treeview(tab, show='headings', height=8)
        self.matrix.grid(row=3, column=0, sticky='nsew')
        horizontal = ttk.Scrollbar(tab, orient='horizontal', command=self.matrix.xview)
        horizontal.grid(row=4, column=0, sticky='ew')
        self.matrix.configure(xscrollcommand=horizontal.set)
        self.result_note = tk.StringVar(value='Sin modelo seleccionado. La aprobación y publicación se realizan desde Administración.')
        tk.Label(tab, textvariable=self.result_note, bg=CARD, fg=MUTED, font=('Segoe UI', 9), anchor='w', wraplength=750, justify='left').grid(row=5, column=0, sticky='ew', pady=12)
        actions = tk.Frame(tab, bg=CARD)
        actions.grid(row=6, column=0, sticky='ew')
        self.folder_button = ttk.Button(actions, text='Abrir carpeta de resultados', command=self.open_result_folder, state='disabled')
        self.folder_button.pack(side='left', padx=(0, 8))
        self.copy_button = ttk.Button(actions, text='Copiar ruta del paquete', command=self.copy_package, state='disabled')
        self.copy_button.pack(side='left')

    @staticmethod
    def set_text(widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('end', text)
        widget.configure(state='disabled')

    def invalidate(self):
        self.audit = None
        if not self.busy:
            self.train_button.configure(state='disabled')
        self.summary.set(f'{len(self.paths)} archivos seleccionados. Audita antes de entrenar.')

    def refresh_files(self):
        self.files_tree.delete(*self.files_tree.get_children())
        for index, file in enumerate(self.paths):
            size = f'{file.stat().st_size / 1024:.0f} KiB' if file.is_file() else 'No existe'
            self.files_tree.insert('', 'end', iid=str(index), values=(file.name, size))
        self.invalidate()

    def load_default(self):
        self.paths = sorted((ROOT / 'datasets').glob(f'dataset-{self.category.get()}-*.json'))
        self.refresh_files()

    def change_category(self, event=None):
        self.version.set(self.new_version())
        self.labels.set('')
        self.load_default()
        self.tabs.select(self.files_tab)
        self.status.set('Módulo cambiado. Revisa los archivos y audita las capturas.')

    def add_files(self):
        selected = filedialog.askopenfilenames(parent=self.root, title='Seleccionar datasets de Kai Señas', initialdir=ROOT / 'datasets' if (ROOT / 'datasets').is_dir() else ROOT, filetypes=[('Capturas JSON', '*.json')])
        for file in selected:
            path = Path(file).resolve()
            if path not in self.paths:
                self.paths.append(path)
        self.refresh_files()

    def remove_files(self):
        indexes = {int(item) for item in self.files_tree.selection()}
        self.paths = [file for index, file in enumerate(self.paths) if index not in indexes]
        self.refresh_files()

    def clear_files(self):
        self.paths = []
        self.refresh_files()

    def selected_labels(self):
        return [value for value in re.split(r'[,;\s]+', self.labels.get().strip()) if value] or None

    def set_busy(self, value, training=False):
        self.busy = value
        self.category_box.configure(state='disabled' if value else 'readonly')
        for widget in self.editable:
            widget.configure(state='disabled' if value else 'normal')
        self.audit_button.configure(state='disabled' if value else 'normal')
        self.train_button.configure(state='normal' if not value and self.audit and self.audit['ready_to_train'] else 'disabled')
        self.cancel_button.configure(state='normal' if value and training else 'disabled')

    def start_audit(self):
        if self.busy:
            return
        if not self.paths:
            messagebox.showinfo('Capturas pendientes', 'Añade los archivos JSON que descargaste de Administración.', parent=self.root)
            return
        args = gui_jobs.command(ROOT, 'audit', self.category.get(), self.paths, self.selected_labels())
        self.set_busy(True)
        self.status.set('Revisando formato, etiquetas, duplicados y separación por sesiones…')
        self.tabs.select(self.files_tab)
        def work():
            try:
                self.events.put(('audit', gui_jobs.run_audit(ROOT, args)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def start_training(self):
        if self.busy or not self.audit or not self.audit['ready_to_train']:
            return
        try:
            version = self.version.get().strip()
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,39}', version):
                raise ValueError('El nombre debe tener hasta 40 letras, números, puntos, guiones o guiones bajos.')
            values = {key: int(variable.get()) for key, variable in [('epochs', self.epochs), ('batch_size', self.batch), ('threads', self.threads)]}
            for key, maximum in [('epochs', 1000), ('batch_size', 1024), ('threads', 16)]:
                if not 1 <= values[key] <= maximum:
                    raise ValueError('Revisa los límites: épocas 1–1000, lote 1–1024, hilos 1–16.')
            destination = ROOT / 'artifacts' / self.category.get() / version
            if destination.exists():
                raise ValueError('Esa versión ya existe. Escribe otro nombre para conservar sus resultados.')
            logs = ROOT / 'artifacts' / 'gui-logs'
            logs.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            cancel_file = ROOT / 'artifacts' / f'.gui-cancel-{token}'
            self.cancel_file = cancel_file
            args = gui_jobs.command(ROOT, 'train', self.category.get(), self.paths, self.selected_labels(), version=version, output=ROOT / 'artifacts', cancel_file=cancel_file, **values)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Revisa la configuración', str(exc), parent=self.root)
            return
        self.set_text(self.log, '')
        self.progress.configure(mode='indeterminate')
        self.progress.start(12)
        self.epoch_text.set('Preparando TensorFlow. La primera época puede tardar…')
        self.status.set('Entrenamiento local en curso. Puedes seguir consultando los resultados anteriores.')
        self.tabs.select(self.training_tab)
        self.set_busy(True, training=True)
        def work():
            try:
                code = gui_jobs.run_training(ROOT, args, lambda kind, value: self.events.put((kind, value)), logs / f'{version}-{token}.log')
                self.events.put(('finished', (code, destination)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
            finally:
                cancel_file.unlink(missing_ok=True)
        threading.Thread(target=work, daemon=True).start()

    def cancel(self):
        if self.busy and self.cancel_file:
            try:
                self.cancel_file.touch(exist_ok=True)
                self.cancel_button.configure(state='disabled')
                self.status.set('Cancelación solicitada. Esperando un punto seguro de TensorFlow…')
            except OSError as exc:
                messagebox.showerror('No se pudo solicitar la cancelación', str(exc), parent=self.root)

    def poll(self):
        try:
            for _ in range(150):
                kind, value = self.events.get_nowait()
                if kind == 'line':
                    self.log.configure(state='normal')
                    self.log.insert('end', value)
                    if int(self.log.index('end-1c').split('.')[0]) > 1200:
                        self.log.delete('1.0', '200.0')
                    self.log.see('end')
                    self.log.configure(state='disabled')
                    epoch = gui_jobs.epoch_from_line(value)
                    if epoch:
                        self.progress.stop()
                        self.progress.configure(mode='determinate', maximum=epoch[1], value=epoch[0])
                        self.epoch_text.set(f'Época {epoch[0]} de {epoch[1]}')
                elif kind == 'audit':
                    self.audit = value
                    self.set_busy(False)
                    self.summary.set(f"{value['unique_samples']} muestras únicas · {len(value['files'])} archivos · {value['duplicates_removed']} duplicadas retiradas")
                    counts = '\n'.join(f"{label}: {count} muestras, {value['files_per_label'][label]} archivos" for label, count in value['samples_per_label'].items())
                    issues = '\n'.join(value['problems']) if value['problems'] else 'Auditoría aprobada. Ya puedes iniciar el entrenamiento.'
                    self.set_text(self.audit_text, counts + '\n\n' + issues)
                    self.status.set('Capturas listas para entrenar. Esto no garantiza precisión con personas nuevas.' if value['ready_to_train'] else 'Hay datos por corregir. Revisa el detalle de la auditoría.')
                elif kind == 'finished':
                    code, destination = value
                    self.progress.stop()
                    self.cancel_file = None
                    self.set_busy(False)
                    status_path = destination / 'status.json'
                    state = json.loads(status_path.read_text(encoding='utf-8')).get('state') if status_path.is_file() else None
                    if code == 0 and state == 'complete':
                        self.progress.configure(mode='determinate', maximum=1, value=1)
                        self.status.set('Entrenamiento completado. Revisa las métricas antes de enviar el paquete a aprobación.')
                        self.load_report(destination / 'training-report.json')
                    elif state == 'cancelled' or code == 130:
                        self.status.set('Ejecución interrumpida o cancelada. No se publicó ningún modelo. Revisa el registro si no solicitaste detenerla.')
                    else:
                        self.status.set('El entrenamiento falló. Consulta el registro y los archivos de diagnóstico de esta versión.')
                    self.epoch_text.set('Proceso finalizado. Usa una versión nueva si necesitas repetirlo.')
                elif kind == 'error':
                    self.progress.stop()
                    self.cancel_file = None
                    self.set_busy(False)
                    self.status.set('No se pudo completar la operación. Revisa el mensaje e inténtalo de nuevo.')
                    messagebox.showerror('Kai Señas', value, parent=self.root)
        except queue.Empty:
            pass
        except Exception as exc:
            self.set_busy(False)
            self.status.set('No se pudo mostrar el resultado. Consulta los archivos de la versión.')
            messagebox.showerror('No se pudo leer el resultado', str(exc), parent=self.root)
        if self.closing and not self.busy:
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def open_report(self):
        file = filedialog.askopenfilename(parent=self.root, title='Abrir training-report.json', initialdir=ROOT / 'artifacts' if (ROOT / 'artifacts').is_dir() else ROOT, filetypes=[('Informe JSON', '*.json')])
        if file:
            try:
                self.load_report(Path(file))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                messagebox.showerror('Informe no válido', str(exc), parent=self.root)

    def load_report(self, file):
        if file.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('El informe supera el tamaño esperado.')
        report = json.loads(file.read_text(encoding='utf-8'))
        labels, matrix = report['matrix_labels'], report['test_confusion_matrix']
        if not isinstance(labels, list) or not 2 <= len(labels) <= 50 or not all(isinstance(label, str) and len(label) <= 40 for label in labels) or not isinstance(matrix, list) or len(matrix) != len(labels) or any(not isinstance(row, list) or len(row) != len(labels) or any(type(value) is not int or value < 0 for value in row) for row in matrix):
            raise ValueError('La matriz y las etiquetas son incompatibles.')
        validation = float(report['metrics']['validation']['accuracy']) * 100
        test = float(report['metrics']['test']['accuracy']) * 100
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in (validation, test)):
            raise ValueError('Las métricas no contienen porcentajes válidos.')
        self.result_summary.set(f"{report['category']} / {report['version']}\nValidación: {validation:.1f} %     Test: {test:.1f} %     Épocas: {report['epochs_run']}")
        self.matrix.delete(*self.matrix.get_children())
        columns = ['real'] + [f'label-{i}' for i in range(len(labels))]
        self.matrix.configure(columns=columns)
        for column, label in zip(columns, ['Real / predicción'] + labels):
            self.matrix.heading(column, text=label)
            self.matrix.column(column, width=130 if column == 'real' else 65, minwidth=50, stretch=True, anchor='center')
        for label, row in zip(labels, matrix):
            self.matrix.insert('', 'end', values=[label, *row])
        self.result_dir = file.parent.resolve()
        status_file = self.result_dir / 'status.json'
        ready = status_file.is_file() and json.loads(status_file.read_text(encoding='utf-8')).get('state') == 'complete' and (self.result_dir / 'model.kai.json').is_file()
        self.folder_button.configure(state='normal')
        self.copy_button.configure(state='normal' if ready else 'disabled')
        self.result_note.set(('Esta ejecución está completa. ' if ready else 'No hay un paquete completo disponible en esta carpeta. ') + 'Revisa especialmente el test y Nulo. Una auditoría aprobada no garantiza que el modelo reconozca bien las señas. Publicar sigue requiriendo al superadmin.')
        self.tabs.select(self.results_tab)

    def open_result_folder(self):
        if self.result_dir and self.result_dir.is_dir():
            os.startfile(str(self.result_dir))

    def copy_package(self):
        if self.result_dir and (self.result_dir / 'model.kai.json').is_file():
            self.root.clipboard_clear()
            self.root.clipboard_append(str(self.result_dir / 'model.kai.json'))
            self.status.set('Ruta copiada. Selecciona ese archivo desde Administración → Modelos de IA.')

    def close(self):
        if self.busy:
            if messagebox.askyesno('Hay un proceso en curso', '¿Solicitar su cancelación y cerrar cuando termine? No se forzará el cierre del proceso.', parent=self.root):
                self.closing = True
                self.cancel()
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        TrainerWindow(root)
        # Ensure a launcher that suppresses its console does not hide the Tk window.
        root.after(150, lambda: (root.deiconify(), root.lift()))
        root.mainloop()
    except Exception:
        messagebox.showerror('No se pudo abrir el entrenador', traceback.format_exc(), parent=root)
        root.destroy()
