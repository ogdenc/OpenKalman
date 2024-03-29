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
  template<coefficients InputCoefficients, coefficients OutputCoefficients, typed_matrix_nestable TransformationMatrix,
      typed_matrix_nestable ... PerturbationTransformationMatrices> requires
    (MatrixTraits<TransformationMatrix>::rows == OutputCoefficients::dimensions) and
    (MatrixTraits<TransformationMatrix>::columns == InputCoefficients::dimensions) and
    ((MatrixTraits<PerturbationTransformationMatrices>::rows == OutputCoefficients::dimensions) and ...) and
    (square_matrix<PerturbationTransformationMatrices> and ...)
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
        typed_matrix<T> and equivalent_to<typename MatrixTraits<T>::RowCoefficients, R> and
          equivalent_to<typename MatrixTraits<T>::ColumnCoefficients, C>>> : std::true_type {};

      template<typename T, typename R, typename C>
      struct is_linear_transformation_input<T, R, C, std::enable_if_t<
        typed_matrix_nestable<T> and (MatrixTraits<T>::rows == R::dimensions) and
        (MatrixTraits<T>::columns == C::dimensions)>> : std::true_type {};
    }
#endif


    /**
     * \internal
     * \brief T is a suitable input to a linear tests constructor.
     * \tparam RowCoefficients The row \ref OpenKalman::coefficients "coefficients" of the linear tests matrix.
     * \tparam ColumnCoefficients The column \ref OpenKalman::coefficients "coefficients" of the linear tests
     * matrix.
     */
#ifdef __cpp_concepts
    template<typename T, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients>
    concept linear_transformation_input =
      (typed_matrix<T> or typed_matrix_nestable<T>) and
      coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and
      (not typed_matrix<T> or (equivalent_to<typename MatrixTraits<T>::RowCoefficients, RowCoefficients> and
          equivalent_to<typename MatrixTraits<T>::ColumnCoefficients, ColumnCoefficients>)) and
      (not typed_matrix_nestable<T> or (MatrixTraits<T>::rows == RowCoefficients::dimensions and
        MatrixTraits<T>::columns == ColumnCoefficients::dimensions));
#else
    template<typename T, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients>
    inline constexpr bool linear_transformation_input =
      coefficients<RowCoefficients> and coefficients<ColumnCoefficients> and
      detail::is_linear_transformation_input<T, RowCoefficients, ColumnCoefficients>::value;
#endif

  } // namespace internal


#ifdef __cpp_concepts
  template<coefficients InputCoefficients, coefficients OutputCoefficients, typed_matrix_nestable TransformationMatrix,
    typed_matrix_nestable ... PerturbationTransformationMatrices> requires
  (MatrixTraits<TransformationMatrix>::rows == OutputCoefficients::dimensions) and
    (MatrixTraits<TransformationMatrix>::columns == InputCoefficients::dimensions) and
    ((MatrixTraits<PerturbationTransformationMatrices>::rows == OutputCoefficients::dimensions) and ...) and
    (square_matrix<PerturbationTransformationMatrices> and ...)
#else
  template<typename InputCoefficients, typename OutputCoefficients, typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
#endif
  struct LinearTransformation
  {

#ifndef __cpp_concepts
    static_assert(coefficients<InputCoefficients>);
    static_assert(coefficients<OutputCoefficients>);
    static_assert(typed_matrix_nestable<TransformationMatrix>);
    static_assert((typed_matrix_nestable<PerturbationTransformationMatrices> and ...));
    static_assert(MatrixTraits<TransformationMatrix>::rows == OutputCoefficients::dimensions);
    static_assert(MatrixTraits<TransformationMatrix>::columns == InputCoefficients::dimensions);
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::rows == OutputCoefficients::dimensions) and ...));
    static_assert((square_matrix<PerturbationTransformationMatrices> and ...));
#endif

  private:

    template<typename Jacobians, typename InputTuple, std::size_t...ints>
    constexpr auto sumprod(Jacobians&& js, InputTuple&& inputs, std::index_sequence<ints...>) const
    {
      return make_self_contained(
        ((std::get<ints>(std::forward<Jacobians>(js)) * std::get<ints>(std::forward<InputTuple>(inputs))) + ...));
    }


    using TransformationMatricesTuple = std::tuple<
      const Matrix<OutputCoefficients, InputCoefficients, self_contained_t<TransformationMatrix>>,
      const Matrix<OutputCoefficients, OutputCoefficients, self_contained_t<PerturbationTransformationMatrices>>...>;

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
    LinearTransformation(T&& mat, Ps&& ... p_mats) noexcept
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
        std::make_index_sequence<sizeof...(Perturbations) + 1>{});
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
        auto id = make_matrix<OutputCoefficients, OutputCoefficients>(MatrixTraits<TransformationMatrix>::identity());
        return std::tuple_cat(transformation_matrices, oin::tuple_replicate<pad_size>(std::move(id)));
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
  template<typed_matrix T, oin::linear_transformation_input<typename MatrixTraits<T>::RowCoefficients> ... Ps>
#else
  template<typename T, typename ... Ps, std::enable_if_t<
    (typed_matrix<T> and ... and oin::linear_transformation_input<Ps, typename MatrixTraits<T>::RowCoefficients>),
    int> = 0>
#endif
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    typename MatrixTraits<T>::ColumnCoefficients,
    typename MatrixTraits<T>::RowCoefficients,
    self_contained_t<nested_matrix_t<T>>,
    self_contained_t<std::conditional_t<typed_matrix<Ps>, nested_matrix_t<Ps>, Ps>>...>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable T, oin::linear_transformation_input<Axes<MatrixTraits<T>::rows>> ... Ps>
#else
  template<typename T, typename ... Ps, std::enable_if_t<
    (typed_matrix_nestable<T> and ... and oin::linear_transformation_input<Ps, Axes<MatrixTraits<T>::rows>>), int> = 0>
#endif
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::rows>,
    self_contained_t<T>,
    self_contained_t<Ps>...>;


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
