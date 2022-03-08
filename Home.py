from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import (StringField,SubmitField)
import joblib

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'

# Now create a WTForm Class
# Lots of fields available:
class InfoForm(FlaskForm):
  
    name = StringField('What\'s your Name?')
    spx = StringField('Enter the SPX Stock Price ?')
    uso = StringField('Enter the United States Oil Fund Price ?')
    slv = StringField('Enter the iShares Silver Trust (SLV) Stock Price ?')
    eur_usd = StringField('What\'s the EUR/USD Price ?')
    submit = SubmitField('Submit')



@app.route('/', methods=['GET', 'POST'])
def index():

    form = InfoForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.
        name = form.name.data
        spx = float(form.spx.data)
        uso = float(form.uso.data)
        slv = float(form.slv.data)
        eur_usd = float(form.eur_usd.data)
        parameters = [spx,uso,slv,eur_usd]
        #the path of the model
        model = joblib.load(r'C:\Users\dell\Desktop\Gold_price\Gold_price.h5')
        y_pred = model.predict([parameters])
        session['result'] = y_pred[0]
        session['name'] = name
        session['spx'] = spx
        session['uso'] = uso
        session['slv'] = slv
        session['eur_usd'] = eur_usd

        return redirect(url_for("price"))


    return render_template('Home.html', form=form)


@app.route('/price')
def price():

    return render_template('the output.html')


if __name__ == '__main__':
    app.run(debug=True)
