//! `--test-shader`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-shader: a user-authored WGSL node through the WHOLE app —
    /// generated WGSL, real pipeline build, real graph.
    ///
    /// Four things only a live run shows, because each one is about the app
    /// reacting to a change rather than about the shader itself:
    /// * the starter shader compiles into a real pipeline;
    /// * its params are full animation channels (keyed value evaluates on
    ///   the curve, authored value untouched) and moving one does NOT move
    ///   the topology hash — a slider must never recompile;
    /// * editing the TEXT does move it, and the new (multi-pass) shader
    ///   comes up;
    /// * a shader that does not compile leaves the comp RENDERING, which is
    ///   the whole premise of typing into a live compositor.
    pub(crate) fn drive_test_shader(&mut self) {
        let Some((stage, mut frames, node)) = self.test.shader else { return };
        frames += 1;
        // A WALL-CLOCK budget: this suite waits for three novel-WGSL
        // pipeline builds (~100-150ms of Metal PSO compile each) and, at
        // stage 3, a full 1.5s. It used to count frames (9000), and the
        // unpaced test loop can spin that many in under the 1.5s wait — so
        // it failed on an idle machine and passed on a busy one. The
        // diagnostic names the state, so a real hang is still legible.
        static T0: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
        if T0.get_or_init(web_time::Instant::now).elapsed() > std::time::Duration::from_secs(60) {
            let d = self.egraph.node(node).and_then(|n| match &n.kind {
                NodeKind::Shader(p) => Some(p.def()),
                _ => None,
            });
            println!(
                "SHADER TEST FAIL: timeout at stage {stage} (ready {}, def ok {:?}, errs {:?})",
                self.last_pipeline_ready,
                d.as_ref().map(|d| d.ok()),
                d.as_ref().map(|d| d.errors.clone()),
            );
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("SHADER TEST FAIL: {m}");
            std::process::exit(1);
        };
        let set_src = |s: &mut Self, src: &str| {
            if let Some(NodeKind::Shader(p)) =
                s.egraph.node_mut(node).map(|n| &mut n.kind)
            {
                p.set_source(src.to_string());
            }
        };
        const TWO_PASS: &str = "//! input: front\n\
//! passes: 2\n\
//! param: float gain = 2.0 [0, 4]\n\
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return tex(front, uv) * P.gain; }\n\
fn pass2(uv: vec2f, frag: vec2f) -> vec4f { return tex(result1, uv); }\n";
        // A Matchbox-style fragment shader, repeated by its preset.
        const GLSL: &str = "#version 120\n\
uniform sampler2D front, adsk_results_pass1;\n\
uniform float adsk_result_w, adsk_result_h, gain;\n\
uniform int adsk_result_execution;\n\
vec2 res = vec2(adsk_result_w, adsk_result_h);\n\
void main() {\n\
  vec2 uv = gl_FragCoord.xy / res;\n\
  gl_FragColor = adsk_result_execution == 0 ? texture2D(front, uv) * gain : texture2D(adsk_results_pass1, uv);\n\
}\n\
<ShaderNodePreset Name=\"Suite\"><Shader Index=\"1\" NbExecutions=\"2\">\
<Uniform Name=\"gain\" Type=\"float\" Default=\"0.5\" DisplayName=\"Gain\"></Uniform>\
</Shader></ShaderNodePreset>\n";
        const BROKEN: &str =
            "//! input: front\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return nope(uv); }\n";
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let id = self.egraph.add(
                    NodeKind::Shader(Default::default()),
                    egui::pos2(320.0, 40.0),
                );
                self.egraph.connect(id, 0, 3);
                self.egraph.view = Some(id);
                self.test.shader = Some((1, frames, id));
            }
            // The starter shader must reach a live pipeline.
            1 if frames > 20 => {
                if !self.last_pipeline_ready {
                    self.test.shader = Some((1, frames, node));
                    return;
                }
                let ch = anim::Channel::Shader(anim::intern("amount"));
                if let Some(n) = self.egraph.node_mut(node) {
                    let c = n.anim.entry(ch).or_default();
                    c.set_key(0, 0.0);
                    c.set_key(10, 1.0);
                }
                let at5 = self
                    .egraph
                    .to_engine_graph_at(Some(5))
                    .nodes
                    .get(&node)
                    .and_then(|k| anim::get(k, ch));
                if at5 != Some(0.5) {
                    fail(format!("keyed shader param evals {at5:?}, want 0.5"));
                }
                let authored =
                    self.egraph.node(node).and_then(|n| anim::get(&n.kind, ch));
                if authored != Some(1.0) {
                    fail(format!("authored {authored:?}, want 1.0 (untouched)"));
                }
                // A param move must not be a topology change.
                let before = self.egraph.to_engine_graph().topology_hash();
                if let Some(n) = self.egraph.node_mut(node) {
                    n.anim.remove(&ch);
                    anim::set(&mut n.kind, ch, 0.5);
                }
                let after = self.egraph.to_engine_graph().topology_hash();
                if before != after {
                    fail("moving a param moved the topology hash".into());
                }
                // Editing the text must.
                set_src(self, TWO_PASS);
                if self.egraph.to_engine_graph().topology_hash() == after {
                    fail("editing the source did not move the topology hash".into());
                }
                self.test.shader = Some((2, frames, node));
            }
            // The two-pass shader must come up too.
            2 if frames > 20 => {
                if !self.last_pipeline_ready {
                    self.test.shader = Some((2, frames, node));
                    return;
                }
                set_src(self, GLSL);
                let def = self.egraph.node(node).and_then(|n| match &n.kind {
                    NodeKind::Shader(p) => Some(p.def()),
                    _ => None,
                });
                match def {
                    Some(d) if d.ok() && d.passes == 2 && d.param("gain").is_some() => {}
                    d => fail(format!("GLSL shader: {:?}", d.map(|d| d.errors.clone()))),
                }
                self.test.shader = Some((4, 0, node));
            }
            // The GLSL shader must reach a live pipeline as well.
            4 if frames > 40 => {
                if !self.last_pipeline_ready {
                    self.test.shader = Some((4, frames, node));
                    return;
                }
                set_src(self, BROKEN);
                if self
                    .egraph
                    .node(node)
                    .is_some_and(|n| matches!(&n.kind, NodeKind::Shader(p) if p.def().ok()))
                {
                    fail("the broken source compiled".into());
                }
                self.test.shader = Some((3, frames, node));
            }
            // ...and a broken one leaves the comp rendering, passing through.
            3 if frames > 20
                && self.test.focus_t.get_or_insert_with(web_time::Instant::now).elapsed()
                    > web_time::Duration::from_millis(1500) =>
            {
                self.test.focus_t = None;
                if !self.last_pipeline_ready {
                    self.test.shader = Some((3, frames, node));
                    return;
                }
                if self.displayed_total < 2 {
                    fail("comp stopped rendering with a broken shader".into());
                }
                println!(
                    "SHADER TEST PASS: starter compiled; param keyed 0->1 evals 0.5 at frame 5 (authored untouched) and moves no topology hash; a text edit does; two-pass shader came up; a repeated Matchbox GLSL shader came up; a broken shader passes through with the comp still rendering"
                );
                std::process::exit(0);
            }
            _ => self.test.shader = Some((stage, frames, node)),
        }
    }
}
