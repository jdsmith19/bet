from datetime import datetime

def generate_html_report(html: str) -> bool:
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	filename = f"results/results_{ timestamp }.html"
	with open(filename, 'w') as f:
		f.write(html)
	return True