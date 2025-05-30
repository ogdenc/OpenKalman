/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMATION_HPP
#define OPENKALMAN_LINEARTRANSFORMATION_HPP


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


  /**
   * \brief A linear tests from one single-column vector to another.
   * \tparam InputCoefficients Coefficient types for the input.
   * \tparam OutputCoefficients Coefficient types for the output.
   * \tparam TransformationMatrix Transformation matrix. It is a native matrix type with rows corresponding to
   * OutputCoefficients and columns corresponding to InputCoefficients.
   * \tparam PerturbationTransformationMatrices Transformation matrices for each potential perturbation term.
   * if the parameter is not given, the tests matrix is assumed to be identity (i.e., it is a translation).
   * It is a native matrix type with both rows and columns corresponding to OutputCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_pattern InputCoefficients, fixed_pattern OutputCoefficients, typed_matrix_nestable TransformationMatrix,
      typed_matrix_nestable ... PerturbationTransformationMatrices> requires
    (index_dimension_of_v<TransformationMatrix, 0> == coordinates::dimension_of_v<OutputCoefficients>) and
    (index_dimension_of_v<TransformationMatrix, 1> == coordinates::dimension_of_v<InputCoefficients>) and
    ((index_dimension_of_v<PerturbationTransformationMatrices, 0> == coordinates::dimension_of_v<OutputCoefficients>) and ...) and
    (square_shaped<PerturbationTransformationMatrices> and ...)
#else
  template<typename InputCoefficients, typename OutputCoefficients, typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
#endif
  struct LinearTransformation;


  namespace internal
  {
#ifdef __cpp_concepts
    template<typename In, typename Out, typename Tm, typename...Pm, std::size_t order> requires (order <= 1)
    struct is_linearized_function<LinearTransformation<In, Out, Tm, Pm...>, order> : std::true_type {};
#else
    template<typename In, typename Out, typename Tm, typename...Pm, std::size_t order>
    struct is_linearized_function<LinearTransformation<In, Out, Tm, Pm...>, order, std::enable_if_t<order <= 1>>
      : std::true_type {};
#endif


#ifndef __cpp_concepts
    namespace detail
    {
      template<typename T, typename R, typename C, typename = void>
      struct is_linear_transformation_input : std::false_type {};

      template<typename T, typename R, typename C>
      struct is_linear_transformation_input<T, R, C, std::enable_if_t<
        typed_matrix<T> and compares_with<vector_space_descriptor_of_t<T, 0>, R>and
          compares_with<vector_space_descriptor_of_t<T, 1>, C>>>: std::true_type {};

      template<typename T, typename R, typename C>
      struct is_linear_transformation_input<T, R, C, std::enable_if_t<
        typed_matrix_nestable<T> and (index_dimension_of<T, 0>::value == coordinates::dimension_of_v<R>) and
        (index_dimension_of<T, 1>::value == coordinates::dimension_of_v<C>)>> : std::true_type {};
    }
#endif


    /**
     * \internal
     * \brief T is a suitable input to a linear tests constructor.
     * \tparam RowCoefficients The row \ref fixed_pattern of the linear tests matrix.
     * \tparam ColumnCoefficients The column \ref fixed_pattern of the linear tests
     * matrix.
     */
#ifdef __cpp_concepts
    template<typename T, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients>
    concept linear_transformation_input =
      (typed_matrix<T> or typed_matrix_nestable<T>) and
      fixed_pattern<RowCoefficients> and fixed_pattern<ColumnCoefficients> and
      (not typed_matrix<T> or (compares_with<vector_space_descriptor_of_t<T, 0>, RowCoefficients>and
          compares_with<vector_space_descriptor_of_t<T, 1>, ColumnCoefficients>)) and
      (not typed_matrix_nestable<T> or (index_dimension_of_v<T, 0> == coordinates::dimension_of_v<RowCoefficients> and
        index_dimension_of_v<T, 1> == coordinates::dimension_of_v<ColumnCoefficients>));
#else
    template<typename T, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients>
    constexpr bool linear_transformation_input =
      fixed_pattern<RowCoefficients> and fixed_pattern<ColumnCoefficients> and
      detail::is_linear_transformation_input<T, RowCoefficients, ColumnCoefficients>::value;
#endif

  } // namespace internal


#ifdef __cpp_concepts
  template<fixed_pattern InputCoefficients, fixed_pattern OutputCoefficients, typed_matrix_nestable TransformationMatrix,
    typed_matrix_nestable ... PerturbationTransformationMatrices> requires
  (index_dimension_of_v<TransformationMatrix, 0> == coordinates::dimension_of_v<OutputCoefficients>) and
    (index_dimension_of_v<TransformationMatrix, 1> == coordinates::dimension_of_v<InputCoefficients>) and
    ((index_dimension_of_v<PerturbationTransformationMatrices, 0> == coordinates::dimension_of_v<OutputCoefficients>) and ...) and
    (square_shaped<PerturbationTransformationMatrices> and ...)
#else
  template<typename InputCoefficients, typename OutputCoefficients, typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
#endif
  struct LinearTransformation
  {

#ifndef __cpp_concepts
    static_assert(fixed_pattern<InputCoefficients>);
    static_assert(fixed_pattern<OutputCoefficients>);
    static_assert(typed_matrix_nestable<TransformationMatrix>);
    static_assert((typed_matrix_nestable<PerturbationTransformationMatrices> and ...));
    static_assert(index_dimension_of_v<TransformationMatrix, 0> == coordinates::dimension_of_v<OutputCoefficients>);
    static_assert(index_dimension_of_v<TransformationMatrix, 1> == coordinates::dimension_of_v<InputCoefficients>);
    static_assert(((index_dimension_of_v<PerturbationTransformationMatrices, 0> == coordinates::dimension_of_v<OutputCoefficients>) and ...));
    static_assert((square_shaped<PerturbationTransformationMatrices> and ...));
#endif

  private:

    template<typename Jacobians, typename InputTuple, std::size_t...ints>
    constexpr auto sumprod(Jacobians&& js, InputTuple&& inputs, std::index_sequence<ints...>) const
    {
      return make_self_contained(
        ((std::get<ints>(std::forward<Jacobians>(js)) * std::get<ints>(std::forward<InputTuple>(inputs))) + ...));
    }


    using TransformationMatricesTuple = std::tuple<
      const Matrix<OutputCoefficients, InputCoefficients, equivalent_self_contained_t<TransformationMatrix>>,
      const Matrix<OutputCoefficients, OutputCoefficients, equivalent_self_contained_t<PerturbationTransformationMatrices>>...>;

    const TransformationMatricesTuple transformation_matrices;

  public:

    /**
     * \brief Constructor.
     */
    LinearTransformation(const TransformationMatrix& mat, const PerturbationTransformationMatrices& ... p_mats)
      : transformation_matrices {mat, p_mats...} {}


    /**
     * \brief General constructor.
     * \tparam T
     * \tparam Ps
     */
#ifdef __cpp_concepts
    template<oin::linear_transformation_input<OutputCoefficients, InputCoefficients> T,
      oin::linear_transformation_input<OutputCoefficients> ... Ps>
#else
    template<typename T, typename ... Ps, std::enable_if_t<
      oin::linear_transformation_input<T, OutputCoefficients, InputCoefficients> and
      (oin::linear_transformation_input<Ps, OutputCoefficients> and ...), int> = 0>
#endif
    LinearTransformation(T&& mat, Ps&& ... p_mats)
      : transformation_matrices {std::forward<T>(mat), std::forward<Ps>(p_mats)...} {}


    /// Applies the tests.
#ifdef __cpp_concepts
    template<transformation_input<InputCoefficients> In, perturbation<OutputCoefficients> ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In, InputCoefficients> and
      (perturbation<Perturbations, OutputCoefficients> and ...), int> = 0>
#endif
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      return sumprod(
        jacobian(in, ps...),
        std::forward_as_tuple(std::forward<In>(in), oin::get_perturbation(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<sizeof...(Perturbations) + 1> {});
    }


    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input<InputCoefficients> In, perturbation<OutputCoefficients> ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In, InputCoefficients> and
      (perturbation<Perturbations, OutputCoefficients> and ...), int> = 0>
#endif
    auto jacobian(const In&, const Perturbations& ...) const
    {
      constexpr auto mat_count = std::tuple_size_v<TransformationMatricesTuple>;

      // If there are more inputs than tests matrices, pad the tuple with extra identity matrices.
      if constexpr(sizeof...(Perturbations) + 1 > mat_count)
      {
        constexpr auto pad_size = sizeof...(Perturbations) + 1 - mat_count;
        auto id = make_vector_space_adapter(make_identity_matrix_like<TransformationMatrix>(), OutputCoefficients{}, OutputCoefficients{});
        return std::tuple_cat(transformation_matrices, oin::tuple_fill<pad_size>(std::move(id)));
      }
      else
      {
        return oin::tuple_slice<0, sizeof...(Perturbations) + 1>(transformation_matrices);
      }
    }

  };


  /*
   * Deduction guides
   */

#ifdef __cpp_concepts
  template<typed_matrix T, oin::linear_transformation_input<vector_space_descriptor_of_t<T, 0>> ... Ps>
#else
  template<typename T, typename ... Ps, std::enable_if_t<
    (typed_matrix<T> and ... and oin::linear_transformation_input<Ps, vector_space_descriptor_of_t<T, 0>>),
    int> = 0>
#endif
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    vector_space_descriptor_of_t<T, 1>,
    vector_space_descriptor_of_t<T, 0>,
    equivalent_self_contained_t<nested_object_of_t<T>>,
    equivalent_self_contained_t<std::conditional_t<typed_matrix<Ps>, nested_object_of_t<Ps>, Ps>>...>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable T, oin::linear_transformation_input<Dimensions<index_dimension_of_v<T, 0>>> ... Ps>
#else
  template<typename T, typename ... Ps, std::enable_if_t<
    (typed_matrix_nestable<T> and ... and oin::linear_transformation_input<Ps, Dimensions<index_dimension_of<T, 0>::value>>), int> = 0>
#endif
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    Dimensions<index_dimension_of_v<T, 1>>,
    Dimensions<index_dimension_of_v<T, 0>>,
    equivalent_self_contained_t<T>,
    equivalent_self_contained_t<Ps>...>;


  /*
   * Traits
   */

  namespace internal
  {
    template<typename InC, typename OutC, typename T, typename ... Ps>
    struct is_linearized_function<LinearTransformation<InC, OutC, T, Ps...>, 0> : std::true_type {};

    template<typename InC, typename OutC, typename T, typename ... Ps>
    struct is_linearized_function<LinearTransformation<InC, OutC, T, Ps...>, 1> : std::true_type {};

  } // namespace internal

}


#endif //OPENKALMAN_LINEARTRANSFORMATION_HPP
