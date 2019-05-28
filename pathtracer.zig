// WIP ported to Zig by Alexej Lesser, based on 
// smallpt, a Path Tracer by Kevin Beason, 2008
// http://www.kevinbeason.com/smallpt/

const std = @import("std");
const jpeg = @import("jpeg_writer.zig");
const print = @import("printf.zig");
//const std.math = @import("std.math");

const multi_threaded = true;

var ray_count : u64 = 0;

const width  = 1024; //2048;
const height = 768; //1536;
const samps = 8;
//const fov: f64 = std.math.pi / 3.0;

const out_filename = "out.jpg";
const out_quality = 100;

const __seed48 = []u64{0xe66d, 0xdeec, 0x5, 0xb };

fn __rand48_step(xi:*[3]u64) u64 {
    var a: u64 = undefined;
    var x: u64 = undefined;
    x = xi[0] | xi[1]<<16 | xi[2]+0<<32;
    a = __seed48[0] | __seed48[1]<<16 | __seed48[2]+0<<32;
    x = (a*%x +% __seed48[3]);
    xi[0] = x;
    xi[1] = x>>16;
    xi[2] = x>>32;
    return x & 0xffffffffffff;
}

fn erand48(s: *[3]u64) f64 {
    const F64AndBits = packed union {
        u: u64,
        f: f64,
    }; 
    var x = F64AndBits{ .u = 0x3ff0000000000000 };
    x.u = x.u | __rand48_step(s)<<4;
    return x.f - 1.0;
}

const RenderContext = struct {
    pixmap: []u8,               // 
    start: u64,               // 
    end: u64,                 // 
    //spheres: []const Sphere,    // 
    //lights: []const Light,    // no pointlights in a pathtracer, done by emission in material
};

const cam = Ray.init(vec3(50.0,52.0,295.6), vec3(0.0,-0.042612,-1.0).normalize()); // new
//const cam = Ray.init(vec3(50.0,52.0,295.6), vec3(0.0,-0.042612,-1.0).normalize()); // new

const spheres = []const Sphere{     //Scene: radius, position, emission, color, material
        
        Sphere{
            .radius = 1e5, 
            .position = vec3(1e5+1.0,40.8,81.6), 
            .emission = vec3(0,0,0),
            .color = vec3(0.75,0.25,0.25),
            .reflection = Refl.DIFF},       //Left
        Sphere{
            .radius = 1e5, 
            .position = vec3(-1e5+99.0,40.8,81.6),
            .emission = vec3(0,0,0),
            .color = vec3(0.25,0.25,0.75),
            .reflection = Refl.DIFF},       //Rght
        Sphere{
            .radius = 1e5, 
            .position = vec3(50.0,40.8,1e5),
            .emission = vec3(0,0,0),
            .color = vec3(0.75,0.75,0.75),
            .reflection = Refl.DIFF},       //Back
        Sphere{
            .radius = 1e5, 
            .position = vec3(50.0,40.8,-1e5+170.0), 
            .emission = vec3(0,0,0),
            .color = vec3(0.25,0.75,0.25),           
            .reflection = Refl.DIFF},       //Frnt
        Sphere{
            .radius = 1e5, 
            .position = vec3(50.0, 1e5,81.6),    
            .emission = vec3(0,0,0),
            .color = vec3(0.75,0.75,0.75),
            .reflection = Refl.DIFF},       //Botm
        Sphere{
            .radius = 1e5, 
            .position = vec3(50.0,-1e5+81.6,81.6),
            .emission = vec3(0,0,0),
            .color = vec3(0.75,0.75,0.75),
            .reflection = Refl.DIFF},       //Top
        Sphere{
            .radius = 16.5,
            .position = vec3(27,16.5,47),       
            .emission = vec3(0,0,0),
            .color = vec3(0.999,0.999,0.999), 
            .reflection = Refl.SPEC},       //Mirr
        Sphere{
            .radius = 16.5,
            .position = vec3(73,16.5,78),       
            .emission = vec3(0,0,0),
            .color = vec3(0.999,0.999,0.999), 
            .reflection = Refl.REFR},       //Glas
        Sphere{
            .radius = 600, 
            .position = vec3(50.0,681.6-0.27,81.6),
            .emission = vec3(12,12,12),
            .color = vec3(0,0,0),
            .reflection = Refl.DIFF}        //Lite
        
    };

fn vec3(x: f64, y: f64, z: f64) Vec3f {
    return Vec3f{ .x = x, .y = y, .z = z };
}

const Vec3f = Vec3(f64);

fn Vec3(comptime T: type) type {
    return struct {
        const Self = @This();

        x: T,
        y: T,
        z: T,

        fn mul(u: Self, v: Self) T {
            return u.x * v.x + u.y * v.y + u.z * v.z;
        }

        fn mult(u: Self, v: Self) Self {
            return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
        }

        fn mulScalar(u: Self, k: T) Self {
            return vec3(u.x * k, u.y * k, u.z * k);
        }

        fn add(u: Self, v: Self) Self {
            return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
        }

        fn addScalar(u: Self, k: T) Self {
            return vec3(u.x + k, u.y + k, u.z + k);
        }

        fn sub(u: Self, v: Self) Self {
            return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
        }

        fn negate(u: Self) Self {
            return vec3(-u.x, -u.y, -u.z);
        }

        fn norm(u: Self) T {
            return std.math.sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
        }

        fn normalize(u: Self) Self {
            return u.mulScalar(1 / u.norm());
        }

        fn cross(u: Vec3f, v: Vec3f) Vec3f {
            return vec3(
                u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x,
            );
        }

        fn dot(u: Vec3f, v: Vec3f) f64 {
            return (
                u.x * v.x +
                u.y * v.y +
                u.z * v.z 
            );
        }
    };
}

const Refl = enum(u2) {         // material types, used in radiance()
    DIFF,
    SPEC,
    REFR,
};

const Ray = struct { 
    origin: Vec3f,
    destination: Vec3f,
    fn init(o: Vec3f, d:Vec3f) Ray {
        return Ray {
            .origin = o,
            .destination = d,
        };
    }
};

const Sphere = struct {
    radius: f64,
    position: Vec3f,
    emission: Vec3f,
    color: Vec3f,
    reflection: Refl,

    fn intersect(self: Sphere, r: Ray) f64 {              // returns distance, 0 if nohit
        var op: Vec3f = Vec3f.sub(self.position,r.origin);  // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        var t:f64 = undefined;
        const eps:f64 = 1e-4;
        var b :f64 = op.dot(r.destination);
        var det:f64 = b*b - op.dot(op) + (self.radius * self.radius) ;
        if (det<0) return 0 else det=std.math.sqrt(det);
        t=b-det;
        if (t>eps) return t else { 
            t=b+det;
            if (t>eps) return t else return 0;
        }
    }
};

fn clamp(x: f64) f64 { return if(x<0) 0 else if (x>1) 1 else x; }
//fn toInt(x: f64) i32 { return pow(clamp(x),1/2.2)*255+0.5; } // TODO impl cast??

fn intersect (r: Ray, t_ptr: *f64, id_ptr: *u32) bool {
    var n: u32 = spheres.len;
    var d: f64 = undefined;
    var inf: f64 = 1e20;
    t_ptr.* = 1e20;
    var i: u32 = 0;
    while (i<n) {
        d = spheres[i].intersect(r);
        if((d!=0.0) and d < t_ptr.* ) {t_ptr.* = d; id_ptr.* = i; }
        i += 1;
    }
    return t_ptr.* < inf;
}

fn mod_sign(b: bool) f64 {
    return if (b) 1.0 else -1.0;
}

fn radiance(r: Ray, depth_ptr: *i32,Xi: *[3]u64) Vec3f {                        // TODO referenzen etc
    ray_count += 1;

    var t:f64 = undefined;                                                      // distance to intersection
    var id:u32 = undefined;                                                     // id of intersected object TODO u8 or more
    if (!intersect(r, &t, &id)) {
        //print._warn("No Hit! {}\n",id);  
        return vec3(0,0,0);                                                     // if miss, return black                            
    }   
    const obj = spheres[id];                                                    // the hit object
    //print._warn("Hit!! {}, Color: ({},{},{})\n",id, obj.color.x,obj.color.y,obj.color.z);

    var x:  Vec3f = Vec3f.add(r.origin,Vec3f.mulScalar(r.destination,t));       // intersection point
    var n:  Vec3f = Vec3f.sub(x,obj.position).normalize();                      // ?
    var nl: Vec3f = if (n.dot(r.destination)<0) n else Vec3f.mulScalar(n,-1.0); // ?
    var f:  Vec3f = obj.color;                                                  // 

    var p: f64 =                            // max component of rgb color value
        if (f.x>f.y and f.x>f.z)
            f.x 
        else
            if (f.y>f.z) 
                f.y
            else f.z;

    depth_ptr.* += 1;
    if (depth_ptr.* >5) {                   // randomly taking emission or color of hit after 6 or more bounces 
        if (erand48(Xi)<p) {
            f=Vec3f.mulScalar(f,(1/p));     // If the color is dark, chanche of emission is higher
        }
        else {
            if (obj.emission.x == 12) {
            return obj.emission;            //R.R.
            }
            else {
            return Vec3f.mult(f,Vec3f.mulScalar(vec3(12,12,12),1.0/@intToFloat(f64,depth_ptr.*)));            // HACKS!!!!!!!!!!!!!
            }
        }
    }
    
    //reflection behaviour-----------------------------------------------------------------------------------------------------------------

    if (obj.reflection == Refl.DIFF){                                                                           // Ideal DIFFUSE reflection
        var r1:     f64 = 2.0 * std.math.pi * erand48(Xi);
        var r2:     f64 = erand48(Xi);
        var r2s:    f64 = std.math.sqrt(r2);
        var w = nl;
        var u = (Vec3f.cross(if (std.math.fabs(w.x)>0.1) vec3(0,1,0) else vec3(1,0,0),w)).normalize();
        var v = Vec3f.cross(w,u);
        var d = (Vec3f.add(Vec3f.add(Vec3f.mulScalar(u,std.math.cos(r1)*r2s), Vec3f.mulScalar(v,std.math.sin(r1)*r2s)), Vec3f.mulScalar(w,std.math.sqrt(1-r2)))).normalize();
        return Vec3f.add(obj.emission,
            Vec3f.mult(f,
                radiance(Ray.init(x,d),depth_ptr,Xi)));
    } 

    else if (obj.reflection == Refl.SPEC)                                                                       // Ideal SPECULAR reflection
        return 
            Vec3f.add(obj.emission,
            Vec3f.mult(f,radiance(Ray.init(x,Vec3f.sub(r.destination,Vec3f.mulScalar(n,2.0*Vec3f.dot(n, r.destination)))),depth_ptr,Xi)));

    var reflRay = Ray.init(x, Vec3f.sub(r.destination,Vec3f.mulScalar(n,2.0*Vec3f.dot(n, r.destination))));     // Ideal dielectric REFRACTION

    //-------------------------------------------------------------------------------------------------------------------------------------

    var into:   bool = Vec3f.dot(n,nl)>0;                // Ray from outside going in?
    var nc:     f64 = 1;
    var nt:     f64 = 1.5 ;
    var nnt:    f64 = if (into) nc/nt else nt/nc;
    var ddn:    f64 = r.destination.dot(nl);
    var cos2t:  f64 = 1-nnt*nnt*(1-ddn*ddn);
    if (cos2t < 0)                                  // Total internal reflection
        return 
        Vec3f.add(obj.emission,
        Vec3f.mult(f,radiance(reflRay,depth_ptr,Xi)));

    var tdir = (
            Vec3f.sub(
                Vec3f.mulScalar(r.destination,nnt),
                Vec3f.mulScalar(
                    n,
                    (
                        (if (into) f64(1.0) else f64(-1.0)) * (ddn*nnt+std.math.sqrt(cos2t))
                    )
                )
            )
        ).normalize();

    var a:      f64 = nt-nc;
    var b:      f64 = nt+nc;
    var R0:     f64 = a*a/(b*b);
    var c:      f64 = 1-(if (into) -ddn else tdir.dot(n));
    var Re:     f64 = R0+(1-R0)*c*c*c*c*c;
    var Tr:     f64 = 1-Re;
    var P:      f64 = 0.25+0.5*Re;
    var RP:     f64 = Re/P;
    var TP:     f64 = Tr/(1-P);

    var rad: Vec3f = undefined;
    if (depth_ptr.*>2) {
        if (erand48(Xi)<P) { // Russian roulette
            rad = Vec3f.mulScalar(radiance(reflRay,depth_ptr,Xi),RP);
        }
        else {
            rad = Vec3f.mulScalar(radiance(Ray.init(x,tdir),depth_ptr,Xi),TP);
        }
    }
    else {
        rad = Vec3f.add(
                Vec3f.mulScalar( radiance(reflRay,depth_ptr,Xi)          , Re),
                Vec3f.mulScalar( radiance(Ray.init(x,tdir),depth_ptr,Xi) , Tr)
            );
    }

    return Vec3f.add(obj.emission,Vec3f.mult(f,rad));
}

fn renderFramebufferSegment(context: RenderContext) void {
    print._warn("SegmentStart: {}, End: {}\n",context.start,context.end);
    var y: u64 = context.start;

    var c:   Vec3f = undefined;
    var cx = vec3(@intToFloat(f64,width)*0.5135/@intToFloat(f64,height),0,0);
    var cy = Vec3f.mulScalar((Vec3f.cross(cx,cam.destination)).normalize(),0.5135);

    while (y < context.end) : (y += 1) {
        var Xi = [3]u64 {0,0,y*y*y};
        var x: u64 = 0;
        while (x < width) : (x += 1) {
            //const x = (2 * (@intToFloat(f32, i) + 0.5) / width - 1) * std.math.tan(fov / 2.0) * width / height;
            //const y = -(2 * (@intToFloat(f32, j) + 0.5) / height - 1) * std.math.tan(fov / 2.0);
            //const direction = vec3(x, y, -1).normalize();
            //var c = castRay(vec3(0, 0, 0), direction, context.spheres, context.lights, 0); OLD
            

            var i   = (height-y-1)*width+x; // notwendig?
            
            c = vec3(0,0,0);
            var sy: u8 = 0;
            while (sy<2){                     // 2x2 subpixel rows
                var sx: u8 = 0;
                while(sx<2){                // 2x2 subpixel cols
                    
                    var r = vec3(0,0,0);
                    var s: u16 = 0;
                    while (s<samps) {
                        var r1: f64 = 2.0*erand48(&Xi);
                        var dx: f64 = if (r1<1.0) std.math.sqrt(r1)-1.0 else 1.0-std.math.sqrt(2.0-r1);
                        var r2: f64 = 2.0*erand48(&Xi);
                        var dy: f64 = if (r2<1.0) std.math.sqrt(r2)-1.0 else 1.0-std.math.sqrt(2.0-r2);

                        var d = Vec3f.add(Vec3f.mulScalar(cx,( ( (@intToFloat(f64,sx)+0.5 + dx)/2.0 + @intToFloat(f64,x))/@intToFloat(f64,width) - 0.5)) ,
                                Vec3f.add(Vec3f.mulScalar(cy,( ( (@intToFloat(f64,sy)+0.5 + dy)/2.0 + @intToFloat(f64,y))/@intToFloat(f64,height) - 0.5)) , cam.destination));

                        var depth: i32 = 0;
                        r = Vec3f.add(r,Vec3f.mulScalar(radiance(Ray.init(Vec3f.add(cam.origin,Vec3f.mulScalar(d,140.0)) ,d.normalize()),&depth,&Xi),1.0/@intToFloat(f64,samps))); // radiance of subpixel sx,sy in point x,y in s samples
                        s+=1;
                    }
                    c = Vec3f.add(c,Vec3f.mulScalar(vec3(clamp(r.x),clamp(r.y),clamp(r.z)),0.25));
                    sx+=1;
                }
                sy+=1;
            }

            //var depth: i32 = 0;
            //var c = radiance(r,&depth,&Xi); OLD

            //print._warn("Point: ({},{}), Color: {}{}{}\n",x,y , c.x,c.y,c.z);

            var max = std.math.max(c.x, std.math.max(c.y, c.z));
            if (max > 1) c = c.mulScalar(1 / max);
            const T = @typeInfo(Vec3f).Struct;
            inline for (T.fields) |field, k| {
                const pixel = @floatToInt(u8, 255.0 * std.math.pow(f64,clamp(@field(c, field.name)),1.0/2.2) + 0.5);
                context.pixmap[3 * (x + y * width) + k] = pixel;
            }
        }
    }
    print._warn("SegmentEnd: {}, End: {}\n",context.start,context.end);
}

fn renderMulti(allocator: *std.mem.Allocator) !void {
    var pixmap = std.ArrayList(u8).init(allocator);
    defer pixmap.deinit();
    try pixmap.resize(3 * width * height);

    const cpu_count = try std.os.cpuCount(allocator);
    const batch_size = height / cpu_count;

    var threads = std.ArrayList(*std.os.Thread).init(allocator);
    defer threads.deinit();

    var j: u64 = 0;
    while (j < height) : (j += batch_size) {
        const context = RenderContext{
            .pixmap = pixmap.toSlice(),
            .start = j,
            .end = j + batch_size,
            //.spheres = spheres,
        };

        try threads.append(try std.os.spawnThread(context, renderFramebufferSegment));
        //renderFramebufferSegment(context);
    }

    for (threads.toSliceConst()) |thread| {
        thread.wait();
    }

    try jpeg.writeToFile(out_filename, width, height, 3, pixmap.toSliceConst(), out_quality);
}

pub fn main() !void {
    //const a_number: i32 = 1234;
    //const a_string = "foobar";
    //print._warn("here is a string: '{}' here is a number: {}\n", a_string, a_number);
    var direct = std.heap.DirectAllocator.init();
    if (multi_threaded) {
        try renderMulti(&direct.allocator);
    } 

    print._warn("Ray Count: {} \n", ray_count);
    //else {
    //    try render(&direct.allocator, spheres, lights);
    //}
}