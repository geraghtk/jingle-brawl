"""The Ring Toss reindeer, built in Blender (the other way to make it: tools/art/reindeer.py is Flux).
Cel-shaded primitives with an outline, like Whack-an-Elf: a sitting reindeer with a body, a scarf,
a battle-ready face (furrowed brows, a smirk, a sticking-plaster), a striped party hat and very wide
antlers, each with a far prong and a tall prong ending in round knobs. One orthographic front camera.

  blender -b --python tools/art/reindeer_blender.py -- out.png [--width 1600]

It prints where the scoring points land in the picture (pixels from the top left): each antler's far
and tall prong tips and the hat's tip. Those go into ringtoss.ts PEGS and RingTossStage's DEER.
"""
import json
import math
import sys
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else 'reindeer-blender.png'
WIDTH = int(argv[argv.index('--width') + 1]) if '--width' in argv else 1600
HEIGHT = int(WIDTH * 0.72)
OUTLINE = (0.09, 0.05, 0.04, 1)


def hexc(h, a=1.0):
    h = h.lstrip('#')
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return (*[c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb], a)


# ---------- scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.film_transparent = True
scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
scene.render.filepath = OUT
scene.view_settings.view_transform = 'Standard'
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
scene.world = world
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = math.pi
sun.data.angle = 0.05
sun.rotation_euler = (math.radians(48), math.radians(-24), math.radians(-28))
scene.collection.objects.link(sun)

cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 6.4
ELEV = math.radians(6)
TARGET = Vector((0, 0, 2.05))
cam.location = TARGET + Vector((0, -20 * math.cos(ELEV), 20 * math.sin(ELEV)))
cam.rotation_euler = (math.pi / 2 - ELEV, 0, 0)
scene.collection.objects.link(cam)
scene.camera = cam
coll = scene.collection

# ---------- materials ----------
_m = {}


def toon(color, shadow=None, spec=0.0):
    key = f'{color}{shadow}{spec}'
    if key in _m:
        return _m[key]
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    diff = nt.nodes.new('ShaderNodeBsdfDiffuse')
    s2r = nt.nodes.new('ShaderNodeShaderToRGB')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.interpolation = 'CONSTANT'
    base = hexc(color)
    ramp.color_ramp.elements[0].color = hexc(shadow) if shadow else tuple(c * 0.55 for c in base[:3]) + (1,)
    ramp.color_ramp.elements[1].position = 0.28
    ramp.color_ramp.elements[1].color = base
    nt.links.new(diff.outputs[0], s2r.inputs[0])
    nt.links.new(s2r.outputs[0], ramp.inputs[0])
    col = ramp.outputs[0]
    if spec:
        gl = nt.nodes.new('ShaderNodeBsdfGlossy')
        gl.inputs['Roughness'].default_value = 0.2
        s2 = nt.nodes.new('ShaderNodeShaderToRGB')
        r2 = nt.nodes.new('ShaderNodeValToRGB')
        r2.color_ramp.interpolation = 'CONSTANT'
        r2.color_ramp.elements[1].position = 0.55
        r2.color_ramp.elements[1].color = (spec, spec, spec, 1)
        nt.links.new(gl.outputs[0], s2.inputs[0])
        nt.links.new(s2.outputs[0], r2.inputs[0])
        add = nt.nodes.new('ShaderNodeMix')
        add.data_type = 'RGBA'
        add.blend_type = 'ADD'
        add.inputs['Factor'].default_value = 1.0
        nt.links.new(col, add.inputs['A'])
        nt.links.new(r2.outputs[0], add.inputs['B'])
        col = add.outputs['Result']
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(col, em.inputs[0])
    nt.links.new(em.outputs[0], out.inputs[0])
    _m[key] = m
    return m


def outline_mat():
    if 'outline' not in _m:
        m = bpy.data.materials.new('outline')
        m.use_nodes = True
        m.use_backface_culling = True
        nt = m.node_tree
        nt.nodes.clear()
        em = nt.nodes.new('ShaderNodeEmission')
        em.inputs[0].default_value = OUTLINE
        nt.links.new(em.outputs[0], nt.nodes.new('ShaderNodeOutputMaterial').inputs[0])
        _m['outline'] = m
    return _m['outline']


def finish(ob, mat, outline=0.03, smooth=True):
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll.objects.link(ob)
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    if smooth and ob.type == 'MESH':
        for p in ob.data.polygons:
            p.use_smooth = True
    if outline:
        ob.data.materials.append(outline_mat())
        mod = ob.modifiers.new('outline', 'SOLIDIFY')
        mod.thickness = outline
        mod.offset = 1
        mod.use_flip_normals = True
        mod.use_rim = False
        mod.material_offset = 1
    return ob


def sphere(mat, loc, scale, rot=(0, 0, 0), outline=0.03):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.scale = scale
    return finish(ob, mat, outline)


def cone(mat, loc, r1, r2, depth, outline=0.03):
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=r1, radius2=r2, depth=depth, location=loc)
    return finish(bpy.context.object, mat, outline)


def torus(mat, loc, R, r, rot=(0, 0, 0), scale=(1, 1, 1), outline=0.025):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=64, minor_segments=16, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.scale = scale
    return finish(ob, mat, outline)


def tube(mat, pts, radius, outline=0.03):
    """A smooth tube through points (x, y, z, radius factor): an antler beam or prong."""
    cu = bpy.data.curves.new('tube', 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 6
    cu.use_fill_caps = True
    sp = cu.splines.new('NURBS')
    sp.points.add(len(pts) - 1)
    for p, (x, y, z, w) in zip(sp.points, pts):
        p.co = (x, y, z, 1)
        p.radius = w
    sp.use_endpoint_u = True
    sp.order_u = 3
    sp.resolution_u = 24
    ob = bpy.data.objects.new('tube', cu)
    coll.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.convert(target='MESH')
    ob = bpy.context.object
    return finish(ob, mat, outline)


# ---------- the reindeer ----------
FUR = toon('#a8673a', '#6e3f20')
FUR_LIGHT = toon('#e9c39a', '#c99b6c')
DARK = toon('#2a1a12', '#140b07')
ANTLER = toon('#c99c6e', '#8a643f')
NOSE = toon('#e3262b', '#9c1015', spec=0.9)
WHITE = toon('#ffffff', '#dde4ee')
PUPIL = toon('#151012', '#000000', spec=0.8)
SCARF = toon('#d7263d', '#8f1426')
SCARF2 = toon('#2f9e6a', '#1d6b47')
HAT_A = toon('#e4322e', '#a51d1b')
HAT_B = toon('#ffc531', '#d18f10')
PLASTER = toon('#f3d2b0', '#d5ab83')
HOOF = toon('#3a2a22', '#1f1611')

# body: sitting, chest forward
sphere(FUR, (0, 0.15, 0.95), (0.78, 0.62, 0.95))
sphere(FUR_LIGHT, (0, -0.32, 1.05), (0.46, 0.25, 0.62), outline=0)
for sx in (-1, 1):                                   # front legs and hooves
    sphere(FUR, (0.36 * sx, -0.22, 0.42), (0.17, 0.17, 0.5))
    sphere(HOOF, (0.38 * sx, -0.32, 0.02), (0.2, 0.2, 0.1))
    sphere(FUR, (0.68 * sx, 0.1, 0.32), (0.32, 0.4, 0.3))   # haunches
# neck and scarf
sphere(FUR, (0, -0.02, 1.85), (0.34, 0.32, 0.48))
torus(SCARF, (0, -0.02, 1.62), 0.37, 0.12, rot=(math.radians(8), 0, 0), scale=(1, 0.95, 1))
sphere(SCARF, (0.22, -0.32, 1.32), (0.12, 0.06, 0.3), rot=(0, math.radians(-12), 0))
sphere(SCARF2, (0.24, -0.34, 1.06), (0.12, 0.06, 0.07), outline=0.02)
# head
HEAD = Vector((0, -0.2, 2.45))
sphere(FUR, HEAD, (0.58, 0.5, 0.56))
sphere(FUR, (0, -0.25, 2.98), (0.16, 0.1, 0.1), rot=(0, math.radians(15), 0), outline=0.02)   # a tuft
sphere(FUR_LIGHT, (0, -0.64, 2.24), (0.36, 0.25, 0.25))            # muzzle
sphere(NOSE, (0, -0.88, 2.33), (0.15, 0.11, 0.12))
for sx in (-1, 1):
    sphere(WHITE, (0.19 * sx, -0.63, 2.58), (0.11, 0.06, 0.13), outline=0.018)
    sphere(PUPIL, (0.16 * sx, -0.685, 2.555), (0.055, 0.03, 0.075), outline=0)
    # furrowed brows: the inner end low (battle face)
    sphere(DARK, (0.2 * sx, -0.64, 2.76), (0.15, 0.05, 0.04), rot=(0, math.radians(-22 * sx), 0), outline=0)
    # ears, out to the side
    sphere(FUR, (0.6 * sx, -0.05, 2.62), (0.3, 0.07, 0.12), rot=(0, math.radians(-20 * sx), 0))
    sphere(toon('#e8a39b', '#b9706a'), (0.6 * sx, -0.11, 2.62), (0.2, 0.03, 0.06), rot=(0, math.radians(-20 * sx), 0), outline=0)
# the smirk (lifted on one side) and a sticking-plaster
bpy.ops.mesh.primitive_torus_add(major_radius=0.17, minor_radius=0.018, major_segments=48, minor_segments=8,
                                 location=(0.02, -0.87, 2.17), rotation=(math.radians(90), math.radians(-8), 0))
mouth = bpy.context.object
import bmesh
bm = bmesh.new(); bm.from_mesh(mouth.data)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.y > -0.05], context='VERTS')   # keep the lower arc
bm.to_mesh(mouth.data); bm.free()
finish(mouth, DARK, outline=0)
sphere(PLASTER, (-0.36, -0.64, 2.36), (0.1, 0.03, 0.045), rot=(0, math.radians(25), 0), outline=0.012)

# the party hat: striped bands up a cone, a gold bobble on top
HAT_BASE, HAT_H, HAT_R = 2.86, 0.9, 0.27
bands = 5
for i in range(bands):
    z0, z1 = HAT_BASE + HAT_H * i / bands, HAT_BASE + HAT_H * (i + 1) / bands
    r0, r1 = HAT_R * (1 - i / bands), HAT_R * (1 - (i + 1) / bands)
    cone(HAT_A if i % 2 == 0 else HAT_B, (0, -0.15, (z0 + z1) / 2), r0, r1, z1 - z0, outline=0.022)
HAT_TIP = Vector((0, -0.15, HAT_BASE + HAT_H))
sphere(HAT_B, HAT_TIP + Vector((0, 0, 0.05)), (0.09, 0.09, 0.09))

# the antlers: a long beam out to the side that curls up into the far prong, and a tall prong from
# its middle; round knobs on both tips (the scoring points)
TIPS = {}
for sx, side in ((-1, 'left'), (1, 'right')):
    beam = [(0.3 * sx, -0.1, 2.85, 1.0), (0.85 * sx, -0.1, 3.05, 0.9), (1.55 * sx, -0.1, 3.05, 0.78),
            (2.05 * sx, -0.1, 3.3, 0.7), (2.2 * sx, -0.1, 3.8, 0.62)]
    tube(ANTLER, beam, 0.1)
    tall = [(1.05 * sx, -0.1, 3.05, 0.85), (1.12 * sx, -0.1, 3.5, 0.75), (1.08 * sx, -0.1, 4.0, 0.65)]
    tube(ANTLER, tall, 0.1)
    for name, (x, z) in (('outer', (2.2 * sx, 3.8)), ('inner', (1.08 * sx, 4.0))):
        sphere(ANTLER, (x, -0.1, z), (0.1, 0.1, 0.1))
        TIPS[f'{side}-{name}'] = Vector((x, -0.1, z))
TIPS['golden'] = HAT_TIP

bpy.ops.render.render(write_still=True)
px = {}
for k, v in TIPS.items():
    c = world_to_camera_view(scene, cam, v)
    px[k] = [round(c.x * WIDTH, 1), round((1 - c.y) * HEIGHT, 1)]
print('POINTS ' + json.dumps({'size': [WIDTH, HEIGHT], 'points': px}))
