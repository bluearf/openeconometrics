/* WebGL2 network renderer. Positions and radii live once in an RGBA32F texture; camera changes update uniforms only. */
(function (root) {
  "use strict";
  const VERTEX = `#version 300 es
precision highp float;
uniform sampler2D u_positions; uniform vec2 u_resolution; uniform vec3 u_camera;
uniform int u_texwidth; uniform float u_ratio; uniform float u_opacity; uniform int u_mode; uniform int u_directed;
in vec2 a_endpoints; in float a_size; in vec4 a_color;
out vec4 v_color; out float v_round; out vec2 v_uv;
vec2 pos(int index) { return texelFetch(u_positions, ivec2(index % u_texwidth, index / u_texwidth), 0).xy; }
vec4 screen(vec2 p) { vec2 s = (p * u_camera.x + u_camera.yz) / u_resolution; return vec4(s.x * 2.0 - 1.0, 1.0 - s.y * 2.0, 0.0, 1.0); }
void main() {
  v_color=vec4(a_color.rgb,a_color.a*u_opacity); v_round=0.0; v_uv=vec2(0.0);
  vec2 p=pos(int(a_endpoints.x));
  if(u_mode==0) { int vertex=gl_VertexID; vec2 corner=vec2(vertex==1||vertex==2||vertex==4?1.0:-1.0,vertex==2||vertex==4||vertex==5?1.0:-1.0); gl_Position=screen(p+corner*a_size); v_uv=corner; v_round=1.0; return; }
  vec2 q=pos(int(a_endpoints.y)); vec2 d=q-p; float len=max(length(d),0.000001); vec2 tangent=d/len; vec2 normal=vec2(-tangent.y,tangent.x);
  float halfwidth=max(a_size*.5,.35/u_camera.x); int vertex=gl_VertexID;
  if(u_mode==2) { float size=max(3.0,4.0/u_camera.x); float targetradius=texelFetch(u_positions,ivec2(int(a_endpoints.y)%u_texwidth,int(a_endpoints.y)/u_texwidth),0).z; vec2 tip=q-tangent*(targetradius+1.0); vec2 point=vertex==0?tip:tip-tangent*size+normal*size*(vertex==1?.6:-.6); gl_Position=screen(point); return; }
  if(u_mode==3) { int triangle=vertex/6; int corner=vertex%6; float t=float(triangle+(corner==1||corner==2||corner==4?1:0))*6.2831853/24.0; float side=(corner==0||corner==1||corner==3)?-1.0:1.0; float radius=max(1.0,texelFetch(u_positions,ivec2(int(a_endpoints.x)%u_texwidth,int(a_endpoints.x)/u_texwidth),0).z*1.6); vec2 center=p+vec2(0.0,-radius); gl_Position=screen(center+vec2(cos(t),sin(t))*(radius+side*halfwidth)); return; }
  bool end=vertex==1||vertex==2||vertex==4; float side=vertex==0||vertex==1||vertex==3?-1.0:1.0;
  gl_Position=screen((end?q:p)+normal*halfwidth*side);
}`;
  const FRAGMENT = `#version 300 es
precision highp float; in vec4 v_color; in float v_round; in vec2 v_uv; out vec4 out_color;
void main() { if(v_round>0.5 && length(v_uv)>1.0) discard; out_color=v_color; }`;
  function rgba(color, alpha) {
    let hex = color.slice(1);
    if (hex.length === 3) hex = hex.replace(/./g, (c) => c + c);
    return [
      parseInt(hex.slice(0, 2), 16) / 255,
      parseInt(hex.slice(2, 4), 16) / 255,
      parseInt(hex.slice(4, 6), 16) / 255,
      alpha,
    ];
  }
  class Renderer {
    constructor(canvas, onLoss, onRestore) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl2", {
        alpha: false,
        antialias: true,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
      });
      if (!this.gl || typeof this.gl.createShader !== "function")
        throw new Error("WebGL2 unavailable");
      this.disposed = false;
      this.lost = false;
      this.onLoss = onLoss;
      this.onRestore = onRestore;
      this.buffers = [];
      this.loss = (event) => {
        event.preventDefault();
        this.lost = true;
        onLoss?.();
      };
      this.restore = () => {
        if (this.disposed) return;
        this.lost = false;
        try {
          this.initialize();
          if (onRestore) onRestore();
          else if (this.data) this.upload(this.data);
        } catch (_) {
          onLoss?.();
        }
      };
      canvas.addEventListener("webglcontextlost", this.loss);
      canvas.addEventListener("webglcontextrestored", this.restore);
      try {
        this.initialize();
      } catch (error) {
        this.dispose();
        throw error;
      }
    }
    initialize() {
      const gl = this.gl;
      const shader = (type, source) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
          gl.deleteShader(s);
          throw new Error("Network shader could not compile");
        }
        return s;
      };
      const vs = shader(gl.VERTEX_SHADER, VERTEX),
        fs = shader(gl.FRAGMENT_SHADER, FRAGMENT);
      this.program = gl.createProgram();
      gl.attachShader(this.program, vs);
      gl.attachShader(this.program, fs);
      gl.linkProgram(this.program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
        throw new Error("Network shader could not link");
      this.locations = {};
      for (const name of [
        "u_positions",
        "u_resolution",
        "u_camera",
        "u_texwidth",
        "u_ratio",
        "u_opacity",
        "u_mode",
        "u_directed",
      ])
        this.locations[name] = gl.getUniformLocation(this.program, name);
      this.attributes = {};
      for (const name of ["a_endpoints", "a_size", "a_color"])
        this.attributes[name] = gl.getAttribLocation(this.program, name);
      this.texture = gl.createTexture();
      this.buffers = [];
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(1, 1, 1, 1);
    }
    upload(data) {
      if (this.disposed || this.lost) return;
      const estimate = data.nodes.length * 44 + data.links.length * 28;
      if (estimate > 64 * 1024 * 1024)
        throw new Error("Network GPU buffer budget exceeds 64 MiB");
      this.data = data;
      const gl = this.gl;
      for (const b of this.buffers) gl.deleteBuffer(b);
      this.buffers = [];
      const size = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      this.textureWidth = Math.min(1024, size);
      const rows = Math.max(
        1,
        Math.ceil(data.nodes.length / this.textureWidth),
      );
      if (rows > size)
        throw new Error("Network position texture exceeds this GPU limit");
      const positions = new Float32Array(this.textureWidth * rows * 4);
      for (let i = 0; i < data.nodes.length; i++) {
        positions[i * 4] = data.positions[i * 2];
        positions[i * 4 + 1] = data.positions[i * 2 + 1];
        positions[i * 4 + 2] = data.radius(i);
      }
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        this.textureWidth,
        rows,
        0,
        gl.RGBA,
        gl.FLOAT,
        positions,
      );
      const nodeData = new Float32Array(data.nodes.length * 7);
      let edgeCount = 0,
        loopCount = 0;
      for (let i = 0; i < data.links.length; i++)
        if (data.edgeVisible(i)) {
          if (data.links[i].a === data.links[i].b) loopCount++;
          else edgeCount++;
        }
      const edgeData = new Float32Array(edgeCount * 7),
        loopData = new Float32Array(loopCount * 7);
      for (let i = 0; i < data.nodes.length; i++) {
        const offset = i * 7;
        nodeData[offset] = i;
        nodeData[offset + 1] = i;
        nodeData[offset + 2] = data.radius(i);
        nodeData.set(
          rgba(
            data.nodeColor(i),
            data.nodeVisible(i) ? (data.nodeActive(i) ? 1 : 0.14) : 0,
          ),
          offset + 3,
        );
      }
      let ei = 0,
        li = 0;
      for (let i = 0; i < data.links.length; i++) {
        if (!data.edgeVisible(i)) continue;
        const e = data.links[i],
          arr = e.a === e.b ? loopData : edgeData,
          offset = (e.a === e.b ? li++ : ei++) * 7;
        arr[offset] = e.a;
        arr[offset + 1] = e.b;
        arr[offset + 2] = data.edgeWidth(i);
        arr.set(rgba(data.edgeColor(i), data.edgeAlpha(i)), offset + 3);
      }
      this.nodeCount = data.nodes.length;
      this.edgeCount = edgeCount;
      this.loopCount = loopCount;
      const buffer = (values) => {
        const b = gl.createBuffer();
        this.buffers.push(b);
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW);
        return b;
      };
      this.nodeBuffer = buffer(nodeData);
      this.edgeBuffer = buffer(edgeData);
      this.loopBuffer = buffer(loopData);
      if (typeof gl.getError === "function" && gl.getError() !== gl.NO_ERROR)
        throw new Error("Network GPU allocation failed");
      this.uploadCount = (this.uploadCount || 0) + 1;
    }
    updatePositions(positions, index) {
      if (this.disposed || this.lost) return;
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      if (index !== undefined)
        gl.texSubImage2D(
          gl.TEXTURE_2D,
          0,
          index % this.textureWidth,
          Math.floor(index / this.textureWidth),
          1,
          1,
          gl.RGBA,
          gl.FLOAT,
          new Float32Array([
            positions[index * 2],
            positions[index * 2 + 1],
            this.data.radius(index),
            0,
          ]),
        );
      else {
        const rows = Math.max(1, Math.ceil(this.nodeCount / this.textureWidth));
        const padded = new Float32Array(rows * this.textureWidth * 4);
        for (let i = 0; i < this.nodeCount; i++) {
          padded[i * 4] = positions[i * 2];
          padded[i * 4 + 1] = positions[i * 2 + 1];
          padded[i * 4 + 2] = this.data.radius(i);
        }
        gl.texSubImage2D(
          gl.TEXTURE_2D,
          0,
          0,
          0,
          this.textureWidth,
          rows,
          gl.RGBA,
          gl.FLOAT,
          padded,
        );
      }
    }
    bind(buffer, instanced) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      for (const [name, count, offset] of [
        ["a_endpoints", 2, 0],
        ["a_size", 1, 8],
        ["a_color", 4, 12],
      ]) {
        const location = this.attributes[name];
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, count, gl.FLOAT, false, 28, offset);
        gl.vertexAttribDivisor(location, instanced ? 1 : 0);
      }
    }
    draw(transform, width, height, ratio, opacity, directed) {
      if (this.disposed || this.lost) return false;
      const gl = this.gl,
        L = this.locations;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(L.u_positions, 0);
      gl.uniform1i(L.u_texwidth, this.textureWidth);
      gl.uniform2f(L.u_resolution, width, height);
      gl.uniform3f(L.u_camera, transform.k, transform.x, transform.y);
      gl.uniform1f(L.u_ratio, ratio);
      gl.uniform1f(L.u_opacity, opacity);
      gl.uniform1i(L.u_directed, directed ? 1 : 0);
      this.bind(this.edgeBuffer, true);
      gl.uniform1i(L.u_mode, 1);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.edgeCount);
      if (directed) {
        gl.uniform1i(L.u_mode, 2);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, this.edgeCount);
      }
      this.bind(this.loopBuffer, true);
      gl.uniform1i(L.u_mode, 3);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 144, this.loopCount);
      this.bind(this.nodeBuffer, true);
      gl.uniform1i(L.u_mode, 0);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.nodeCount);
      return true;
    }
    dispose() {
      if (this.disposed) return;
      this.disposed = true;
      this.canvas.removeEventListener("webglcontextlost", this.loss);
      this.canvas.removeEventListener("webglcontextrestored", this.restore);
      if (!this.lost) {
        for (const b of this.buffers) this.gl.deleteBuffer(b);
        if (this.texture) this.gl.deleteTexture(this.texture);
        if (this.program) this.gl.deleteProgram(this.program);
      }
      this.buffers = [];
      this.data = null;
    }
  }
  root.OpenEconNetworkWebGL = { Renderer, rgba };
})(typeof window !== "undefined" ? window : globalThis);
