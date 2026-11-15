import { PathLayer, type PathLayerProps } from "@deck.gl/layers";
import type { Accessor, DefaultProps } from "@deck.gl/core";

/**
 * PathLayer that draws dashes moving along each path, used for water flowing through pipes.
 *
 * Each path vertex carries its distance from the water source (`getDistances`). The fragment
 * shader keeps only the parts where `fract((distance - time) / spacing)` is inside the dash,
 * so dashes travel from low to high distance as `time` increases. Modeled on deck.gl's
 * TripsLayer (same shader hooks), but repeating instead of a single trail.
 */
type FlowLayerExtraProps<D> = {
  /** Cumulative distance from the source at each path vertex, in world units. */
  getDistances?: Accessor<D, number[]>;
  /** Animation clock, in world units travelled. */
  time?: number;
  /** Distance between dash starts, in world units. */
  spacing?: number;
  /** Fraction of each spacing that is drawn (0-1). */
  dashRatio?: number;
};

export type FlowLayerProps<D> = PathLayerProps<D> & FlowLayerExtraProps<D>;

const defaultProps: DefaultProps<FlowLayerProps<unknown>> = {
  getDistances: { type: "accessor", value: (d: unknown) => (d as { distances: number[] }).distances },
  time: { type: "number", value: 0 },
  spacing: { type: "number", value: 50, min: 0 },
  dashRatio: { type: "number", value: 0.35, min: 0, max: 1 },
};

export default class FlowLayer<D = unknown> extends PathLayer<D, FlowLayerExtraProps<D>> {
  static layerName = "FlowLayer";
  static defaultProps = defaultProps;

  getShaders() {
    const shaders = super.getShaders();
    shaders.inject = {
      "vs:#decl": `\
in float instanceDistances;
in float instanceNextDistances;
out float vFlowDist;
`,
      "vs:#main-end": `\
vFlowDist = instanceDistances + (instanceNextDistances - instanceDistances) * vPathPosition.y / vPathLength;
`,
      "fs:#decl": `\
uniform float flowTime;
uniform float flowSpacing;
uniform float flowDashRatio;
in float vFlowDist;
`,
      "fs:#main-start": `\
float flowPhase = fract((vFlowDist - flowTime) / flowSpacing);
if (flowPhase > flowDashRatio) {
  discard;
}
`,
      // Fade each dash toward its tail so it reads as moving water rather than a dashed line.
      // This hook runs in its own function, so the phase is recomputed rather than shared with main().
      "fs:DECKGL_FILTER_COLOR": `\
color.a *= smoothstep(0.0, flowDashRatio, fract((vFlowDist - flowTime) / flowSpacing));
`,
    };
    return shaders;
  }

  initializeState() {
    super.initializeState();
    this.getAttributeManager()!.addInstanced({
      distances: {
        size: 1,
        accessor: "getDistances",
        shaderAttributes: {
          instanceDistances: { vertexOffset: 0 },
          instanceNextDistances: { vertexOffset: 1 },
        },
      },
    });
  }

  draw(params: { uniforms: Record<string, unknown> }) {
    const { time, spacing, dashRatio } = this.props;
    params.uniforms = { ...params.uniforms, flowTime: time, flowSpacing: spacing, flowDashRatio: dashRatio };
    super.draw(params);
  }
}
