"""Whack-an-Elf art: cel-shaded elf, golden elf, Grinch (and their bonked faces), the snowy hole
(back + front lip, so a pop-up sinks into it) and a candy-cane mallet. Built from primitives,
rendered with Eevee to transparent WebP.

  blender -b --python tools/art/whack.py -- web/public/art/whack [--size 384]

Every character renders with the hole's camera, so its layer lines up with the hole's; the
clip line (where a pop-up disappears into the hole) is printed as a % of the image height.
"""
import math
import sys
import bmesh
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else 'whack-out'
SIZE = int(argv[argv.index('--size') + 1]) if '--size' in argv else 384

OUTLINE = (0.10, 0.06, 0.12, 1)


def hexc(h, a=1.0):
    h = h.lstrip('#')
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, a)


# ---------- scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = SIZE * 2
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
sun.rotation_euler = (math.radians(50), math.radians(-28), math.radians(-30))
scene.collection.objects.link(sun)

cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 4.0
ELEV = math.radians(24)
TARGET = Vector((0, 0, 0.95))
cam.location = TARGET + Vector((0, -12 * math.cos(ELEV), 12 * math.sin(ELEV)))
cam.rotation_euler = (math.pi / 2 - ELEV, 0, 0)
scene.collection.objects.link(cam)
scene.camera = cam


# ---------- materials ----------
_mats = {}


def toon(color, shadow=None, spec=0.0, name=None):
    """Two-band cel shading + optional hard specular, emitted so colours stay flat."""
    key = name or f'{color}{shadow}{spec}'
    if key in _mats:
        return _mats[key]
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
    sh = hexc(shadow) if shadow else tuple(c * 0.55 for c in base[:3]) + (1,)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = sh
    ramp.color_ramp.elements[1].position = 0.28
    ramp.color_ramp.elements[1].color = base
    nt.links.new(diff.outputs[0], s2r.inputs[0])
    nt.links.new(s2r.outputs[0], ramp.inputs[0])
    col = ramp.outputs[0]
    if spec:
        gl = nt.nodes.new('ShaderNodeBsdfGlossy')
        gl.inputs['Roughness'].default_value = 0.22
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
    _mats[key] = m
    return m


def flat(color):
    key = 'flat' + color
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs[0].default_value = hexc(color)
    nt.links.new(em.outputs[0], nt.nodes.new('ShaderNodeOutputMaterial').inputs[0])
    _mats[key] = m
    return m


def outline_mat():
    if 'outline' in _mats:
        return _mats['outline']
    m = bpy.data.materials.new('outline')
    m.use_nodes = True
    m.use_backface_culling = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs[0].default_value = OUTLINE
    nt.links.new(em.outputs[0], nt.nodes.new('ShaderNodeOutputMaterial').inputs[0])
    _mats['outline'] = m
    return m


# ---------- mesh helpers ----------
def finish(ob, mat, coll, outline=0.035, smooth=True):
    for c in ob.users_collection:
        c.objects.unlink(ob)
    coll.objects.link(ob)
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


def sphere(coll, mat, loc, scale, outline=0.035, rot=(0, 0, 0), seg=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=seg // 2, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.scale = scale
    return finish(ob, mat, coll, outline)


def cone(coll, mat, loc, r1, depth, rot=(0, 0, 0), r2=0.0, outline=0.035):
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=r1, radius2=r2, depth=depth, location=loc, rotation=rot)
    return finish(bpy.context.object, mat, coll, outline)


def cyl(coll, mat, loc, r, depth, rot=(0, 0, 0), outline=0.035):
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=r, depth=depth, location=loc, rotation=rot)
    return finish(bpy.context.object, mat, coll, outline, smooth=False)


def torus(coll, mat, loc, R, r, rot=(0, 0, 0), arc=None, outline=0.03, scale=(1, 1, 1)):
    """A torus in the XZ plane facing the camera; arc=(a0,a1) degrees keeps only that part
    (0 = +x, 90 = up)."""
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=64, minor_segments=16)
    ob = bpy.context.object
    if arc:
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        a0, a1 = arc
        kill = []
        for v in bm.verts:
            a = math.degrees(math.atan2(v.co.y, v.co.x)) % 360
            lo, hi = a0 % 360, a1 % 360
            inside = lo <= a <= hi if lo <= hi else (a >= lo or a <= hi)
            if not inside:
                kill.append(v)
        bmesh.ops.delete(bm, geom=kill, context='VERTS')
        bm.to_mesh(ob.data)
        bm.free()
    # torus is built in XY; stand it up to face -Y (the camera)
    ob.rotation_euler = (math.pi / 2 + rot[0], rot[1], rot[2])
    ob.location = loc
    ob.scale = scale
    return finish(ob, mat, coll, outline)


def star(coll, mat, loc, r=0.16, rot=0.0, points=4, inner=0.38, depth=0.05):
    bm = bmesh.new()
    verts = []
    for i in range(points * 2):
        a = math.pi / 2 + rot + i * math.pi / points
        rr = r if i % 2 == 0 else r * inner
        verts.append(bm.verts.new((rr * math.cos(a), 0, rr * math.sin(a))))
    f = bm.faces.new(verts)
    ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    bmesh.ops.translate(bm, vec=(0, depth, 0), verts=[e for e in ext['geom'] if isinstance(e, bmesh.types.BMVert)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new('star')
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('star', me)
    coll.objects.link(ob)
    ob.location = loc
    return finish(ob, mat, coll, 0.025, smooth=False)


def on_face(c, rx, ry, rz, x, z, inset=0.97):
    """Point on the front (-Y) of an ellipsoid head centred at c."""
    k = max(0.0, 1 - (x / rx) ** 2 - ((z - c.z) / rz) ** 2)
    return Vector((c.x + x, c.y - ry * math.sqrt(k) * inset, z))


def new_coll(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c


# ---------- the hole ----------
SNOW = toon('#f7fbff', '#b8cbe8', name='snow')
HOLE_Z = 0.05


def mound(coll, front_only=False):
    m = torus(coll, SNOW, (0, 0, 0), 1.22, 0.42, rot=(-math.pi / 2, 0, 0), scale=(1, 1, 0.6), outline=0.035)
    if front_only:
        bm = bmesh.new()
        bm.from_mesh(m.data)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if (m.matrix_world @ v.co).y > 0.02], context='VERTS')
        bm.to_mesh(m.data)
        bm.free()
    lumps = [(-1.25, -0.75, 0.05, 0.36), (1.38, -0.42, 0.02, 0.3), (0.55, -1.42, 0.0, 0.26), (-0.9, 1.05, 0.05, 0.3)]
    for x, y, z, r in lumps:
        if front_only and y > 0:
            continue
        sphere(coll, SNOW, (x, y, z), (r * 1.3, r, r * 0.8))
    return m


hole_back = new_coll('hole_back')
mound(hole_back)
# the pit: a dark disc with a lighter ring of earth just under the snow
cyl(hole_back, toon('#2a1b3d', '#1a1027', name='pit'), (0, 0, HOLE_Z), 0.92, 0.04, outline=0)
cyl(hole_back, toon('#120a1c', name='pitdeep'), (0, 0.06, HOLE_Z + 0.025), 0.68, 0.02, outline=0)
# sparkly bits of snow
for x, y, z in [(-1.55, -0.2, 0.22), (1.1, -1.0, 0.25), (0.2, 1.3, 0.3)]:
    star(hole_back, flat('#ffffff'), (x, y - 0.2, z), r=0.09, depth=0.02)

hole_front = new_coll('hole_front')
mound(hole_front, front_only=True)


# ---------- characters ----------
HEAD_C = Vector((0, 0, 1.12))


def elf(coll, pal, bonk=False):
    rx, ry, rz = 0.62, 0.56, 0.58
    c = HEAD_C
    skin = toon(pal['skin'], pal['skin_sh'], spec=pal.get('spec', 0), name=pal['name'] + 'skin')
    # body, collar, mittens
    sphere(coll, toon(pal['coat'], pal['coat_sh'], spec=pal.get('spec', 0), name=pal['name'] + 'coat'), (0, 0.05, c.z - 0.95), (0.62, 0.5, 0.62))
    torus(coll, toon(pal['trim'], pal['trim_sh'], name=pal['name'] + 'trim'), (0, 0.02, c.z - 0.5), 0.36, 0.1, rot=(-math.pi / 2, 0, 0), scale=(1.1, 1, 0.8))
    for s in (-1, 1):
        sphere(coll, toon(pal['trim'], pal['trim_sh'], name=pal['name'] + 'trim'), (s * 0.66, -0.25, c.z - 0.55), (0.17, 0.15, 0.17))
    sphere(coll, skin, c, (rx, ry, rz))
    # ears
    for s in (-1, 1):
        cone(coll, skin, (s * 0.66, 0.02, c.z + 0.08), 0.16, 0.5, rot=(0, s * math.radians(72), 0))
    # hair fringe under the brim
    hair = toon(pal['hair'], name=pal['name'] + 'hair')
    for x in (-0.36, -0.12, 0.12, 0.36):
        sphere(coll, hair, on_face(c, rx, ry, rz, x, c.z + 0.34, 0.86), (0.15, 0.11, 0.12), outline=0.025)
    # hat: brim, bent cone, bell
    hatm = toon(pal['hat'], pal['hat_sh'], spec=pal.get('spec', 0), name=pal['name'] + 'hat')
    tilt = math.radians(-18 if not bonk else -34)
    torus(coll, toon(pal['brim'], pal['brim_sh'], name=pal['name'] + 'brim'), (0, 0, c.z + 0.42), 0.55, 0.13, rot=(-math.pi / 2, tilt * 0.25, 0), scale=(1, 1, 0.9))
    sphere(coll, hatm, (0, 0, c.z + 0.42), (0.55, 0.52, 0.32))
    base = Vector((0, 0.02, c.z + 0.45))
    tip_len = 1.0
    cone(coll, hatm, base + Vector((math.sin(-tilt) * tip_len / 2, 0, math.cos(tilt) * tip_len / 2)), 0.5, tip_len, rot=(0, tilt, 0))
    tip = base + Vector((math.sin(-tilt) * tip_len, 0, math.cos(tilt) * tip_len))
    sphere(coll, toon(pal['bell'], pal['bell_sh'], spec=0.5, name=pal['name'] + 'bell'), tip, (0.13, 0.13, 0.13))
    # face
    for s in (-1, 1):
        sphere(coll, toon(pal['cheek'], name=pal['name'] + 'cheek'), on_face(c, rx, ry, rz, s * 0.33, c.z - 0.12, 0.93), (0.12, 0.05, 0.08), outline=0)
    sphere(coll, skin, on_face(c, rx, ry, rz, 0, c.z - 0.04, 0.96), (0.07, 0.06, 0.06), outline=0.02)
    ink = toon('#24152b', name='ink')
    if not bonk:
        for s in (-1, 1):
            e = on_face(c, rx, ry, rz, s * 0.22, c.z + 0.06, 0.95)
            sphere(coll, ink, e, (0.1, 0.05, 0.14), outline=0)
            sphere(coll, flat('#ffffff'), e + Vector((0.03, -0.05, 0.05)), (0.04, 0.02, 0.04), outline=0)
            sphere(coll, flat('#ffffff'), e + Vector((-0.035, -0.05, -0.05)), (0.018, 0.01, 0.018), outline=0)
        torus(coll, ink, on_face(c, rx, ry, rz, 0, c.z - 0.13, 0.97), 0.17, 0.028, arc=(200, 340), outline=0)
    else:
        for s in (-1, 1):
            e = on_face(c, rx, ry, rz, s * 0.2, c.z + 0.08, 0.93)
            for a in (45, -45):
                cyl(coll, ink, e, 0.025, 0.2, rot=(math.pi / 2, math.radians(a), 0), outline=0)
        torus(coll, ink, on_face(c, rx, ry, rz, 0, c.z - 0.2, 0.96), 0.08, 0.03, outline=0, scale=(1, 1, 1.3))
        dizzy(coll, c.z + 0.95)


def dizzy(coll, z):
    gold = toon('#ffd23f', '#e09a12', spec=0.4, name='dizzy')
    for i, (x, dz) in enumerate(((-0.8, 0.0), (0.0, 0.32), (0.8, 0.05))):
        star(coll, gold, (x, -0.9, z + 0.45 + dz), r=0.17, rot=0.3 * i, points=5, inner=0.45)


def grinch(coll, bonk=False):
    c = HEAD_C + Vector((0, 0, 0.02))
    rx, ry, rz = 0.68, 0.56, 0.6
    fur = toon('#7cc242', '#4b8a22', name='gfur')
    dark = toon('#3d6b18', name='gdark')
    sphere(coll, toon('#e2324a', '#a3172e', name='gsuit'), (0, 0.05, c.z - 0.95), (0.62, 0.5, 0.62))
    torus(coll, toon('#ffffff', '#c9d6ea', name='gtrim'), (0, 0.02, c.z - 0.5), 0.36, 0.11, rot=(-math.pi / 2, 0, 0), scale=(1.1, 1, 0.8))
    for s in (-1, 1):
        sphere(coll, fur, (s * 0.66, -0.25, c.z - 0.55), (0.17, 0.15, 0.17))
    sphere(coll, fur, c, (rx, ry, rz))
    # cheek tufts
    for s in (-1, 1):
        for dz, ang in ((-0.05, 100), (-0.25, 125)):
            cone(coll, fur, (s * 0.62, 0, c.z + dz), 0.13, 0.36, rot=(0, s * math.radians(ang), 0))
    # brows, eyes
    for s in (-1, 1):
        e = on_face(c, rx, ry, rz, s * 0.22, c.z + 0.08, 0.92)
        sphere(coll, toon('#f6e04a', '#d0b21d', name='geye'), e, (0.15, 0.05, 0.11), outline=0.022)
        ink = toon('#24152b', name='ink')
        if not bonk:
            sphere(coll, toon('#c0262f', name='gpupil'), e + Vector((-s * 0.03, -0.045, -0.01)), (0.055, 0.02, 0.06), outline=0)
            sphere(coll, dark, on_face(c, rx, ry, rz, s * 0.24, c.z + 0.25, 0.94), (0.2, 0.06, 0.06),
                   rot=(0, s * math.radians(-22), 0), outline=0)
        else:
            for a in (45, -45):
                cyl(coll, ink, e + Vector((0, -0.04, 0)), 0.022, 0.17, rot=(math.pi / 2, math.radians(a), 0), outline=0)
    sphere(coll, dark, on_face(c, rx, ry, rz, 0, c.z - 0.06, 0.96), (0.08, 0.05, 0.05), outline=0)
    ink = toon('#24152b', name='ink')
    if not bonk:
        # the sly grin, one corner up
        torus(coll, ink, on_face(c, rx, ry, rz, 0.02, c.z - 0.12, 0.96), 0.3, 0.032, arc=(195, 345), rot=(0, math.radians(-10), 0), outline=0, scale=(1, 1, 0.6))
    else:
        torus(coll, ink, on_face(c, rx, ry, rz, 0, c.z - 0.24, 0.96), 0.12, 0.03, arc=(20, 160), outline=0)
    # Santa hat, flopping to the side
    red = toon('#e2324a', '#a3172e', name='gsuit')
    white = toon('#ffffff', '#c9d6ea', name='gtrim')
    torus(coll, white, (0, 0, c.z + 0.45), 0.58, 0.15, rot=(-math.pi / 2, math.radians(6), 0))
    sphere(coll, red, (0, 0, c.z + 0.47), (0.58, 0.55, 0.3))
    tilt = math.radians(-50 if not bonk else -70)
    base = Vector((0, 0.02, c.z + 0.52))
    L = 0.95
    cone(coll, red, base + Vector((-math.sin(tilt) * L / 2, 0, math.cos(tilt) * L / 2)), 0.54, L, rot=(0, tilt, 0))
    sphere(coll, white, base + Vector((-math.sin(tilt) * L, 0, math.cos(tilt) * L)), (0.16, 0.16, 0.16))
    if bonk:
        dizzy(coll, c.z + 0.95)


ELF = dict(name='elf', skin='#ffd2b0', skin_sh='#e8a184', coat='#2fb35c', coat_sh='#1d7d3d',
           trim='#e2324a', trim_sh='#a3172e', hair='#8a4b2a', hat='#2fb35c', hat_sh='#1d7d3d',
           brim='#e2324a', brim_sh='#a3172e', bell='#ffd23f', bell_sh='#e09a12', cheek='#ff8fa3')
GOLD = dict(name='gold', skin='#ffe08a', skin_sh='#e0a53a', coat='#ffc93c', coat_sh='#c98a1b',
            trim='#fff1b8', trim_sh='#e0b04a', hair='#c98a1b', hat='#ffc93c', hat_sh='#c98a1b',
            brim='#fff1b8', brim_sh='#e0b04a', bell='#ffffff', bell_sh='#ffe08a', cheek='#ffb35c', spec=0.55)

chars = {}
for key, build in [('elf', lambda c: elf(c, ELF)), ('elf-bonk', lambda c: elf(c, ELF, bonk=True)),
                   ('gold', lambda c: elf(c, GOLD)), ('gold-bonk', lambda c: elf(c, GOLD, bonk=True)),
                   ('grinch', lambda c: grinch(c)), ('grinch-bonk', lambda c: grinch(c, bonk=True))]:
    coll = new_coll(key)
    build(coll)
    chars[key] = coll
# the golden elf sparkles
for x, z, r in [(-0.95, 2.05, 0.18), (0.95, 1.55, 0.13), (-0.85, 0.9, 0.1)]:
    star(chars['gold'], flat('#fffbe0'), (x, -0.9, z), r=r)

mallet = new_coll('mallet')
# candy-cane mallet: red head with white bands, striped handle
head = Vector((0, 0, 2.15))
cyl(mallet, toon('#e2324a', '#a3172e', spec=0.5, name='mred'), head, 0.42, 1.25, rot=(0, math.pi / 2, 0))
for x in (-0.5, 0.5):
    cyl(mallet, toon('#ffffff', '#c9d6ea', name='gtrim'), head + Vector((x, 0, 0)), 0.435, 0.16, rot=(0, math.pi / 2, 0))
for x in (-0.66, 0.66):
    cyl(mallet, toon('#2fb35c', '#1d7d3d', name='mgreen'), head + Vector((x, 0, 0)), 0.32, 0.08, rot=(0, math.pi / 2, 0))
for i in range(9):
    z = head.z - 0.45 - i * 0.2
    cyl(mallet, toon('#ffffff' if i % 2 else '#e2324a', name=f'mh{i % 2}'), (0, 0, z), 0.1, 0.2, outline=0.03)


# ---------- render ----------
def show_only(*names):
    for c in scene.collection.children:
        c.hide_render = c.name not in names


def render(name, *colls):
    show_only(*colls)
    scene.render.filepath = f'{OUT}/{name}.png'
    bpy.ops.render.render(write_still=True)


render('hole-back', 'hole_back')
render('hole-front', 'hole_front')
for k in chars:
    render(k, k)
pivot = bpy.data.objects.new('pivot', None)
scene.collection.objects.link(pivot)
for ob in bpy.data.collections['mallet'].objects:
    ob.parent = pivot
pivot.rotation_euler = (math.radians(-20), 0, math.radians(28))
cam.location = Vector((0, -12, 1.4))
cam.rotation_euler = (math.pi / 2, 0, 0)
render('mallet', 'mallet')

# where a pop-up disappears: just under the top edge of the front lip, centre column
img = bpy.data.images.load(f'{OUT}/hole-front.png')
W, H = img.size
px = img.pixels[:]
col = [px[((H - 1 - y) * W + W // 2) * 4 + 3] for y in range(H)]  # top-down alpha
top = next(y for y, a in enumerate(col) if a > 0.5)
bottom = max(y for y, a in enumerate(col) if a > 0.5)
print(f'CLIP_LINE_PCT={(top + (bottom - top) * 0.3) / H * 100:.1f}')
