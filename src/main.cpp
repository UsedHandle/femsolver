#include <deal.ii/base/quadrature_lib.h>
#include <deal.ii/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.ii/base/tensor.h>
#include <deal.II/base/numbers.h>

#include <deal.ii/lac/block_vector.h>
#include <deal.ii/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <limits>
#include <fstream>
#include <iostream>


using namespace dealii;

const double pi = numbers::PI;
constexpr double nu = 0.1;

template<int dim>
class SpecificBodyForce : public TensorFunction<1, dim> {
public:
	SpecificBodyForce() : TensorFunction<1, dim>() {}
	virtual Tensor<1, dim> value(const Point<dim>& p) const override {
		const double& x = p[0];
		const double& y = p[1];
		double tmp;
		Tensor<1, dim> value;
		
		value[0] =
			(1. - 2. * x) * y * (1. - y)
			 -nu * 2. * pi * pi * ((tmp = std::cos(pi * x)) * tmp - 3. * (tmp = std::sin(pi * x)) * tmp) * std::sin(pi * y) * std::cos(pi * y)
			+ pi * (tmp = std::sin(pi * x)) * tmp * tmp * (tmp = std::sin(pi * y)) * tmp * std::cos(pi * x);
		value[1] =
			(1. - 2. * y) * x * (1. - x)
			+ nu * 2. * pi * pi * ((tmp = std::cos(pi * y)) * tmp - 3. * (tmp = std::sin(pi * y)) * tmp) * std::sin(pi * x) * std::cos(pi * x)
			+ pi * (tmp = std::sin(pi * y)) * tmp * tmp * (tmp = std::sin(pi * x)) * tmp * std::cos(pi * y);
		
		return value;
	}
};

template<int dim>
class ExactVelocity : public TensorFunction<1, dim> {
public:
	ExactVelocity() : TensorFunction<1, dim>() {}
	virtual Tensor<1, dim> value(const Point<dim>& p) const override {
		Assert(dim == 2, ExcNotImplemented());
		double tmp;
		const double& x = p[0];
		const double& y = p[1];
		Tensor<1, dim> value;
		value[0] = (tmp = std::sin(pi * p[0])) * tmp * std::sin(pi * p[1]) * std::cos(pi * p[1]);
		value[1] = (tmp = -std::sin(pi * p[1])) * tmp * std::sin(pi * p[0]) * std::cos(pi * p[0]);

		return value;
	}
};


template<int dim>
class ExactPressure : public Function<dim> {
public:
	ExactPressure() : Function<dim>(1) {}
	virtual double value(const Point<dim>& p, const unsigned int component = 0) const override {
		(void)component;
		Assert(component == 0,
			ExcMessage("Invalid operation for scalar function"));
		return p[0] * p[1] * (1. - p[0])*(1. - p[1]);

	}
};

template<int dim>
class ExactSolution : public Function<dim> {
public:
	ExactSolution() : Function<dim>(dim+1) {}
	virtual double value(const Point<dim>& p, const unsigned int component = 0) const override {
		if (component < dim)
			return velocity.value(p)[component];
		return
			pressure.value(p);
	}
private:
	ExactVelocity<dim> velocity;
	ExactPressure<dim> pressure;
};

template<int dim>
struct InnerPreconditioner;

template<>
struct InnerPreconditioner<2> {
	using type = SparseDirectUMFPACK;
};

template<>
struct InnerPreconditioner<3> {
	using type = SparseILU<double>;
};

template<int dim>
class NavierStokes {
public:
	NavierStokes();
	void run();
private:
	void setup_system();
	void assemble_matrices();
	void solve_time_step();
	void output_results();

	Triangulation<dim> triangulation;
	const FESystem<dim> fe;
	DoFHandler<dim> dof_handler;

	AffineConstraints<double> constraints;

	BlockSparsityPattern sparsity_pattern;
	BlockSparseMatrix<double> system_matrix;
	
	
	BlockVector<double> solution;
	BlockVector<double> system_rhs;


	double time_step;
	double time;
	unsigned int timestep_number;
};

template<int dim>
NavierStokes<dim>::NavierStokes() :
	fe(FE_Q<dim>(2)^dim, FE_Q<dim>(1)),
	dof_handler(triangulation),
	time_step(1. / 128.),
	time(0.),
	timestep_number(0)
{}

template<int dim>
void NavierStokes<dim>::setup_system() {
	GridGenerator::hyper_cube(triangulation, 0, 1);
	triangulation.refine_global(4);

	dof_handler.distribute_dofs(fe);

	std::vector<unsigned int> block_component(dim + 1, 0);
	block_component[dim] = 1;
	DoFRenumbering::component_wise(dof_handler, block_component);

	const std::vector<types::global_dof_index> dofs_per_block =
		DoFTools::count_dofs_per_fe_block(dof_handler);
	const types::global_dof_index n_u = dofs_per_block[0];
	const types::global_dof_index n_p = dofs_per_block[1];

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();

	std::cout << "Number of active cells: " << triangulation.n_active_cells()
		<< std::endl
		<< "Total number of cells: " << triangulation.n_cells()
		<< std::endl
		<< "Number of degrees of freedom: " << dof_handler.n_dofs()
		<< " (" << n_u << '+' << n_p << ')' << std::endl;

	BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);

	solution.reinit(dofs_per_block);
	system_rhs.reinit(dofs_per_block);

	solution = 0;
}

template<int dim>
void NavierStokes<dim>::assemble_matrices() {
	const QGauss<dim> quadrature_formula(3);
	const QGauss<dim - 1> face_quadrature_formula(3);

	FEValues<dim> fe_values(
		fe,
		quadrature_formula,
		update_values | update_gradients | update_quadrature_points |
		update_JxW_values
	);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points = quadrature_formula.size();
	const unsigned int n_q_face_points = face_quadrature_formula.size();

	FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> local_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	SpecificBodyForce<dim> specific_body_force;
	std::vector<Tensor<1, dim>> specific_body_force_values(n_q_points);
	std::vector<Tensor<1, dim>> velocity_values(n_q_points);

	const FEValuesExtractors::Vector velocities(0);
	const FEValuesExtractors::Scalar pressure(dim);

	std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
	std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
	std::vector<double> div_phi_u(dofs_per_cell);
	std::vector<double> phi_p(dofs_per_cell);
	
	std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

	for (const auto& cell : dof_handler.active_cell_iterators()) {
		fe_values.reinit(cell);

		local_matrix = 0;
		local_rhs = 0;

		specific_body_force.value_list(fe_values.get_quadrature_points(), specific_body_force_values);
		fe_values[velocities].get_function_values(solution, velocity_values);
		for (unsigned int q = 0; q < n_q_points; ++q) {
			for (unsigned int k = 0; k < dofs_per_cell; ++k) {
				phi_u[k] = fe_values[velocities].value(k, q);
				grad_phi_u[k] = fe_values[velocities].gradient(k, q);
				div_phi_u[k] = fe_values[velocities].divergence(k, q);
				phi_p[k] = fe_values[pressure].value(k, q);
				grad_phi_p[k] = fe_values[pressure].gradient(k, q);
			}

			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
				
					local_matrix(i, j) +=
						((grad_phi_u[j] * velocity_values[q]) * phi_u[i] +
							nu * scalar_product(grad_phi_u[i], grad_phi_u[j]) -
							div_phi_u[i] * phi_p[j] -
							phi_p[i] * div_phi_u[j]) * fe_values.JxW(q);
				

					
				}
				local_rhs(i) +=
					specific_body_force_values[q] * phi_u[i] * fe_values.JxW(q);
			}
		}

		cell->get_dof_indices(local_dof_indices);
		constraints.distribute_local_to_global(
			local_matrix,
			local_rhs,
			local_dof_indices,
			system_matrix,
			system_rhs
		);
		
		std::map<types::global_dof_index, double> boundary_values;
		VectorTools::interpolate_boundary_values(dof_handler,
			types::boundary_id(0),
			ExactSolution<dim>(),
			boundary_values);
		MatrixTools::apply_boundary_values(boundary_values,
			system_matrix,
			solution,
			system_rhs);
	}
}

template<int dim>
void NavierStokes<dim>::solve_time_step() {
	double initial_defect = std::nan("");
	double current_defect = std::numeric_limits<double>::infinity();
	unsigned int iter = 1;
	do {
		assemble_matrices();
		std::cout << "Assembled Matrices, iteration: " << iter++ << std::endl;
		BlockVector<double> newton_rhs = system_rhs;
		newton_rhs *= -1.;
		system_matrix.vmult_add(newton_rhs, solution);
		BlockVector<double> defect;

		SparseDirectUMFPACK solver;
		solver.initialize(system_matrix);

		solver.vmult(defect, newton_rhs);
		if (std::isnan(initial_defect))
			initial_defect = defect.l2_norm();
		else
			current_defect = defect.l2_norm();
		solution -= defect;
		std::cout << current_defect << std::endl;
		
	} while (current_defect / initial_defect > 1e-6 && iter <= 50);

}

template<int dim>
void NavierStokes<dim>::output_results() {
	std::vector<std::string> solution_names(dim, "velocity");
	solution_names.emplace_back("pressure");

	for (int i = 0; i < dim; ++i)
		solution_names[i] += "_" + std::to_string(i);

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);


	BlockVector<double> exact_solution;
	exact_solution.reinit(solution);
	std::vector<std::string> exact_solution_names(dim, "exact_velocity");
	exact_solution_names.emplace_back("exact_pressure");
	for (int i = 0; i < dim; ++i)
		exact_solution_names[i] += "_" + std::to_string(i);
	VectorTools::interpolate(dof_handler, ExactSolution<dim>(), exact_solution);

	data_out.add_data_vector(exact_solution, exact_solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);

	data_out.build_patches();

	std::ofstream output("solution.vtk");
	data_out.write_vtk(output);
}

template<int dim>
void NavierStokes<dim>::run() {
	setup_system();
	solve_time_step();
	std::cout << "Done solving" << std::endl;
	output_results();
}

int main() {
	NavierStokes<2> navier_stokes;
	navier_stokes.run();
	return 0;
}