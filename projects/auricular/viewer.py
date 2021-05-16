import trimesh
import trimesh.viewer

import pyglet

def basicCallback(scene):
    if scene.new_index != scene.current_index:
        scene.current_index = scene.new_index
        mesh = trimesh.load_mesh(scene.sample[scene.current_index]['filename'])
        scene.delete_geometry("specimen")
        scene.add_geometry(mesh, "specimen")

def showSample(sample, callback=basicCallback, init_index=0):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.load_mesh(sample[0]['filename']), "specimen")
    scene.current_index = init_index
    scene.new_index = init_index
    scene.sample = sample
    window = trimesh.viewer.SceneViewer(
        scene=scene,
        callback=callback,
        callback_period=0.1,
        start_loop=False,
    )

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            window.close()
        if symbol == pyglet.window.key.N:
            scene.new_index = min(len(scene.sample) - 1, scene.new_index + 1)
        if symbol == pyglet.window.key.P:
            scene.new_index = max(0, scene.new_index - 1)

    pyglet.app.run()
