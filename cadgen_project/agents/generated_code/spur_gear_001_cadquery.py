"""
Spur Gear Design
Detailed CAD plan for creating a spur gear with 24 teeth, a pitch diameter of 60mm, and a 5mm bore hole, suitable for 3D printing or CNC milling.
"""

import cadquery as cq
import math

def create_gear_teeth(num_teeth, module, pressure_angle, thickness):
    # Simplified gear tooth profile
    # In real implementation, use involute curve equations
    pitch_radius = num_teeth * module / 2
    outer_radius = pitch_radius + module
    tooth_width = module * 0.5  # Adjust as needed

    teeth = None
    for i in range(num_teeth):
        angle = (360 / num_teeth) * i
        tooth = (cq.Workplane("XY")
                 .moveTo(pitch_radius, 0)
                 .lineTo(outer_radius, tooth_width/2)
                 .lineTo(outer_radius, -tooth_width/2)
                 .close()
                 .extrude(thickness)
                 .rotate((0,0,0), (0,0,1), angle))
        teeth = teeth.union(tooth) if teeth else tooth
    return teeth


# Main construction
pitch_diameter = 60.0
gear_thickness = 8.0
bore_radius = 2.5
number_of_teeth = 24
module = 2.5
pressure_angle = 20
chamfer_radius = 0.5
pitch_circle_radius = 30

result = cq.Workplane("XY").circle(pitch_circle_radius).extrude(gear_thickness)

bore_circle = cq.Workplane("XY").circle(bore_radius)
result = result.cut(bore_circle.extrude(gear_thickness))

gear_profile = create_gear_teeth(number_of_teeth, module, pressure_angle, gear_thickness)
result = result.cut(gear_profile)

result = result.edges("|Z").fillet(chamfer_radius)


cq.exporters.export(result, 'spur_gear.stl')
