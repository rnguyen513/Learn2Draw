import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import http.server
import socketserver
import os
import json
import threading
import argparse

parser = argparse.ArgumentParser(description="Live Colored Curve Optimizer")
parser.add_argument("--whole", action="store_true", help="Draw the whole image instead of isolating edges")
args, _ = parser.parse_known_args()

# CONFIG - edit this
IMAGE_FILE = "r.jpg"
IMAGE_SIZE = (500,500) # resolution for image to be downscaled
FILE_PATH = "desmos_high_res_curves.txt" # file that stores all bezier curves
PORT = 3000
NUM_CURVES = 500
EPOCHS = 800

def extract_target_data(image_path, size=IMAGE_SIZE, threshold=0.5, whole_image=False):
    img_gray = Image.open(image_path).convert('L').resize(size)
    img_array = np.array(img_gray) / 255.0
    
    img_color = Image.open(image_path).convert('RGB').resize(size)
    color_array = np.array(img_color)
    
    if whole_image:
        print("Mode: WHOLE IMAGE (mosaic)")
        y_idx, x_idx = np.where(img_array >= 0.0) 
        if len(y_idx) > 4000:
            choices = np.random.choice(len(y_idx), 4000, replace=False)
            y_idx, x_idx = y_idx[choices], x_idx[choices]
    else:
        print("Mode: ISOLATE SUBJECT (edge sketch)")
        y_idx, x_idx = np.where(img_array < threshold)
    
    x_norm = (x_idx / size[0]) * 2.0 - 1.0
    y_norm = -((y_idx / size[1]) * 2.0 - 1.0) 
    
    target_points = torch.tensor(np.column_stack((x_norm, y_norm)), dtype=torch.float32)
    return target_points, color_array

target_points, color_array = extract_target_data(IMAGE_FILE, whole_image=args.whole)

class MultiBezierLearner(nn.Module):
    def __init__(self, num_curves=50):
        super().__init__()
        self.num_curves = num_curves
        
        centers = torch.rand(num_curves, 1, 2) * 2.0 - 1.0
        offsets = torch.randn(num_curves, 4, 2) * 0.05 
        
        self.control_points = nn.Parameter(centers + offsets)
        
    def forward(self, t):
        t = t.view(1, -1, 1)
        P0 = self.control_points[:, 0:1, :]
        P1 = self.control_points[:, 1:2, :]
        P2 = self.control_points[:, 2:3, :]
        P3 = self.control_points[:, 3:4, :]
        curves = ((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3
        return curves.view(-1, 2)

def chamfer_loss(preds, targets, control_points=None, penalty_weight=0.03):
    dists = torch.cdist(preds, targets) 
    loss_preds = torch.min(dists, dim=1)[0].mean()
    loss_targets = torch.min(dists, dim=0)[0].mean()
    chamfer = loss_preds + loss_targets
    
    if control_points is not None:
        P0 = control_points[:, 0, :]
        P1 = control_points[:, 1, :]
        P2 = control_points[:, 2, :]
        P3 = control_points[:, 3, :]
        lengths = torch.norm(P1-P0, dim=1) + torch.norm(P2-P1, dim=1) + torch.norm(P3-P2, dim=1)
        length_penalty = lengths.mean()
        return chamfer + (penalty_weight * length_penalty)
        
    return chamfer

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Colored Curve Optimizer</title>
    <script src="https://www.desmos.com/api/v1.8/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
    <style>
        html, body { margin: 0; padding: 0; width: 100%; height: 100%; background: #111; font-family: sans-serif; overflow: hidden; }
        #calculator { width: calc(100% - 380px); height: 100vh; float: left; }
        #status-panel { 
            position: absolute; top: 20px; left: 20px; z-index: 100; 
            background: rgba(0, 0, 0, 0.8); color: #0f0; 
            padding: 10px 15px; border-radius: 5px; border: 1px solid #333;
        }
        #equation-sidebar {
            width: 380px; height: 100vh; float: right; background: #1a1a1a; 
            color: #ddd; overflow-y: auto; padding: 20px; box-sizing: border-box;
            border-left: 2px solid #333; box-shadow: -5px 0 15px rgba(0,0,0,0.5);
        }
        .eq-item { font-family: monospace; font-size: 10px; margin-bottom: 15px; padding-left: 10px; word-break: break-all; line-height: 1.4; }
        h3 { margin-top: 0; color: #fff; border-bottom: 1px solid #444; padding-bottom: 10px; font-size: 16px; }
    </style>
</head>
<body>
    <div id="status-panel">Waiting for PyTorch data...</div>
    <div id="calculator"></div>
    <div id="equation-sidebar">
        <h3>Live Math</h3>
        <div id="eq-list"><i>Waiting for curves to deploy...</i></div>
    </div>

    <script>
        var elt = document.getElementById('calculator');
        var calculator = Desmos.GraphingCalculator(elt, { 
            expressions: false, settingsMenu: false, zoomButtons: true,
            showGrid: false, showAxes: false
        });
        calculator.setMathBounds({ left: -1.2, right: 1.2, bottom: -1.2, top: 1.2 });

        var lastModifiedTime = 0;

        function fetchLiveCurves() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    if (data.mtime > lastModifiedTime) {
                        lastModifiedTime = data.mtime;
                        
                        // Notice: calculator.setBlank() is completely gone!

                        var sidebarHTML = "";
                        var newExpressions = data.equations.map((line, index) => {
                            var parts = line.split('|');
                            var eq = parts[0];
                            var color = parts[1];
                            
                            if (index < 20) {
                                sidebarHTML += `<div class="eq-item" style="border-left: 4px solid ${color};">
                                    <strong style="color:${color}; font-size: 12px;">Curve ${index + 1}</strong><br>${eq}
                                </div>`;
                            }

                            var randomThickness = Math.random() * 3.0 + 10.0;

                            return { id: 'curve_' + index, latex: eq, color: color, lines: true, lineWidth: randomThickness };
                        });
                        
                        // 1. Remove old curves manually instead of wiping the whole board
                        var newIds = newExpressions.map(e => e.id);
                        calculator.getExpressions().forEach(expr => {
                            if (!newIds.includes(expr.id)) {
                                calculator.removeExpression({ id: expr.id });
                            }
                        });
                        
                        // 2. Update the graph in-place
                        calculator.setExpressions(newExpressions);
                        
                        document.getElementById('status-panel').innerText = 'Live: ' + data.equations.length + ' colored curves';
                        if (sidebarHTML !== "") {
                            document.getElementById('eq-list').innerHTML = sidebarHTML;
                        }
                    }
                })
                .catch(err => {});
        }
        setInterval(fetchLiveCurves, 1000);
        fetchLiveCurves();
    </script>
</body>
</html>
"""

class LiveDesmosHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == '/data':
            if not os.path.exists(FILE_PATH):
                self.send_response(404); self.end_headers(); return
            mtime = os.path.getmtime(FILE_PATH)
            with open(FILE_PATH, 'r') as f: lines = [line.strip() for line in f if line.strip()]
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"mtime": mtime, "equations": lines}).encode('utf-8'))
        else:
            super().do_GET()

def start_server():
    with socketserver.TCPServer(("", PORT), LiveDesmosHandler) as httpd:
        httpd.serve_forever()

threading.Thread(target=start_server, daemon=True).start()
print(f"--> Live Viewer running at http://localhost:{PORT}")

model = MultiBezierLearner(num_curves=NUM_CURVES)
optimizer = optim.Adam(model.parameters(), lr=0.05)

num_points_per_curve = 20
t_vals = torch.linspace(0, 1, num_points_per_curve) 

deploy_all_by_epoch = EPOCHS // 2
add_curve_every = max(1, deploy_all_by_epoch // NUM_CURVES)

for epoch in range(EPOCHS + 1):
    active_curves = min(NUM_CURVES, 1 + (epoch // add_curve_every))
    
    optimizer.zero_grad()
    pred_points = model(t_vals)
    
    active_preds = pred_points[:active_curves * num_points_per_curve]
    active_cps = model.control_points[:active_curves]
    
    loss = chamfer_loss(active_preds, target_points, control_points=active_cps, penalty_weight=0.03)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Active Curves: {active_curves}/{NUM_CURVES} | Loss: {loss.item():.4f}")
        
        P = model.control_points.detach().numpy()
        
        with open(FILE_PATH, "w") as f:
            for i in range(active_curves):
                x_eq = f"(1-t)^3 * {P[i,0,0]:.4f} + 3(1-t)^2 * t * {P[i,1,0]:.4f} + 3(1-t) * t^2 * {P[i,2,0]:.4f} + t^3 * {P[i,3,0]:.4f}"
                y_eq = f"(1-t)^3 * {P[i,0,1]:.4f} + 3(1-t)^2 * t * {P[i,1,1]:.4f} + 3(1-t) * t^2 * {P[i,2,1]:.4f} + t^3 * {P[i,3,1]:.4f}"
                
                mid_x = 0.125*P[i,0,0] + 0.375*P[i,1,0] + 0.375*P[i,2,0] + 0.125*P[i,3,0]
                mid_y = 0.125*P[i,0,1] + 0.375*P[i,1,1] + 0.375*P[i,2,1] + 0.125*P[i,3,1]
                
                px = int(np.clip((mid_x + 1.0) / 2.0 * IMAGE_SIZE[0], 0, IMAGE_SIZE[0] - 1))
                py = int(np.clip((1.0 - mid_y) / 2.0 * IMAGE_SIZE[1], 0, IMAGE_SIZE[1] - 1))
                
                r, g, b = color_array[py, px]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                
                f.write(f"({x_eq}, {y_eq})|{hex_color}\n")

print("\ndrawing complete")
print(f"the web server is still running")

import time
try:
    while True:
        time.sleep(1) 
except KeyboardInterrupt:
    print("\nshutting down.")