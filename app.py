"""
Flask Application for Product Recommendations
Implements item-based collaborative filtering with sentiment analysis
"""

from flask import Flask, render_template, request, flash
import os

from model import initialize_engine, get_recommendations

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize the recommendation engine at startup
print("=" * 50)
print("Starting Flask Application...")
print("=" * 50)

try:
    # Get paths relative to the app directory
    csv_path = os.path.join(os.path.dirname(__file__), 'Reviews.csv')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    initialize_engine(csv_path=csv_path, models_dir=models_dir)
    print("=" * 50)
    print("Application ready to serve requests!")
    print("=" * 50)
except Exception as e:
    print(f"ERROR: Failed to initialize recommendation engine: {e}")
    print("Application may not work correctly.")


@app.route('/')
def index():
    """Render the home page with username input form"""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """Process username and return recommendations"""
    username = request.form.get('username', '').strip()
    
    # Validate input
    if not username:
        flash('Please enter a username.', 'error')
        return render_template('index.html'), 400
    
    # Get recommendations
    result = get_recommendations(
        user=username,
        top_n_recommend=20,
        top_n_final=5
    )
    
    if not result['success']:
        flash(result['message'], 'error')
        return render_template('index.html', username=username), 404
    
    # Render results page
    return render_template(
        'results.html',
        username=result['user'],
        recommendations=result['recommendations']
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    flash('An internal error occurred. Please try again.', 'error')
    return render_template('index.html'), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

