from flask import Flask, render_template
import subprocess
import sys
import os

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/courses')
def courses():
    return render_template('courses.html')

@app.route('/revision')
def revision():
    return render_template('revision.html')

@app.route('/teacher')
def teacher():
    return render_template('teacher.html')

@app.route('/chimie')
def chimie():
    return render_template('chimie.html')

@app.route('/french')
def french():
    return render_template('french.html')

@app.route('/notes')
def notes():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    keyboard_script = os.path.join(current_dir, 'virtual_keyboard.py')
    
    try:
        # Run the virtual keyboard script as a subprocess
        subprocess.Popen([sys.executable, keyboard_script])
        return render_template('notes.html')
    except Exception as e:
        return f"Error launching keyboard: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)




