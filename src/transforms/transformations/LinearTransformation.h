/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMATION_H
#define OPENKALMAN_LINEARTRANSFORMATION_H


namespace OpenKalman
{

  /**
   * @brief A linear transformation from one single-column vector to another.
   *
   * @tparam InputCoefficients Coefficient types for the input.
   * @tparam OutputCoefficients Coefficient types for the output.
   * @tparam TransformationMatrix Transformation matrix. It is a native matrix type with rows corresponding to
   * OutputCoefficients and columns corresponding to InputCoefficients.
   * @tparam PerturbationTransformationMatrices Transformation matrices for each potential perturbation term.
   * if the parameter is not given, the transformation matrix is assumed to be identity (i.e., it is a translation).
   * It is a native matrix type with both rows and columns corresponding to OutputCoefficients.
   */
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransformation;


  template<typename In, typename Out, typename Tm, typename...Pm, std::size_t order>
  struct is_linearized_function<LinearTransformation<In, Out, Tm, Pm...>, order,
    std::enable_if_t<order <= 2>> : std::true_type {};


  namespace internal
  {
    namespace detail
    {
      template<std::size_t begin, typename T, std::size_t... I>
      constexpr auto tuple_slice_impl(T&& t, std::index_sequence<I...>)
      {
        return std::forward_as_tuple(std::get<begin + I>(std::forward<T>(t))...);
      }
    }

    /// Return a subset of a tuple, given an index range.
    template<std::size_t index1, std::size_t index2, typename T>
    constexpr auto tuple_slice(T&& t)
    {
      static_assert(index1 <= index2, "Index range is invalid");
      static_assert(index2 <= std::tuple_size_v<std::decay_t<T>>, "Index is out of bounds");
      return detail::tuple_slice_impl<index1>(std::forward<T>(t), std::make_index_sequence<index2 - index1>());
    }

    /// Implementation of the main linear transformation function.
    template<typename TransformationMatricesTuple>
    struct FunctionImpl
    {
    protected:
      const TransformationMatricesTuple& transformation_matrices;

      template<std::size_t i, typename Input>
      constexpr decltype(auto) sumprod_term(Input&& input) const
      {
        if constexpr(i < std::tuple_size_v<TransformationMatricesTuple>)
        {
          return std::get<i>(transformation_matrices) * std::forward<Input>(input);
        }
        else
          // If there is no corresponding transformation matrix, treat the transformation matrix as identity.
          return std::forward<Input>(input);
      }

      template<typename InputTuple, std::size_t...ints>
      constexpr auto sumprod(InputTuple&& inputs, std::index_sequence<ints...>) const
      {
        return strict((sumprod_term<ints>(std::get<ints>(std::forward<InputTuple>(inputs))) + ...));
      }

    public:
      FunctionImpl(const TransformationMatricesTuple& trans) : transformation_matrices(trans) {}

      template<typename...Inputs>
      auto operator()(Inputs&&...inputs) const
      {
        return sumprod(std::tuple {std::forward<Inputs>(inputs)...}, std::make_index_sequence<sizeof...(Inputs)>{});
      }
    };


    /// Implementation of the linear Jacobian function.
    template<typename TransformationMatricesTuple>
    struct JacobianImpl
    {
    protected:
      const TransformationMatricesTuple& transformation_matrices;

    public:
      JacobianImpl(const TransformationMatricesTuple& trans) : transformation_matrices(trans) {}

      template<typename...Inputs>
      auto operator()(Inputs&&...inputs) const
      {
        constexpr auto mat_count = std::tuple_size_v<TransformationMatricesTuple>;

        // If there are more inputs than transformation matrices, pad the list with extra identity matrices.
        if constexpr(sizeof...(Inputs) > mat_count)
        {
          constexpr auto pad_size = sizeof...(Inputs) - mat_count;
          using TransformationMatrix = std::tuple_element_t<0, TransformationMatricesTuple>;
          constexpr auto dim = MatrixTraits<TransformationMatrix>::dimension;
          using PStrict = typename MatrixTraits<TransformationMatrix>::template StrictMatrix<dim, dim>;
          auto id = MatrixTraits<PStrict>::identity();
          return std::tuple_cat(transformation_matrices, internal::tuple_replicate<pad_size>(id));
        }
        else
        {
          return internal::tuple_slice<0, sizeof...(Inputs)>(transformation_matrices);
        }
      }
    };

  }


  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransformation : Transformation<InputCoefficients, OutputCoefficients,
    internal::FunctionImpl<std::tuple<TypedMatrix<OutputCoefficients, InputCoefficients, TransformationMatrix>,
      TypedMatrix<OutputCoefficients, OutputCoefficients, PerturbationTransformationMatrices>...>>,
    internal::JacobianImpl<std::tuple<TypedMatrix<OutputCoefficients, InputCoefficients, TransformationMatrix>,
      TypedMatrix<OutputCoefficients, OutputCoefficients, PerturbationTransformationMatrices>...>>>
  {
    static_assert(is_typed_matrix_base_v<TransformationMatrix>);
    static_assert(std::conjunction_v<is_typed_matrix_base<PerturbationTransformationMatrices>...>);
    static_assert(MatrixTraits<TransformationMatrix>::dimension == OutputCoefficients::size);
    static_assert(MatrixTraits<TransformationMatrix>::columns == InputCoefficients::size);
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::dimension == OutputCoefficients::size) and ...));
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::columns == OutputCoefficients::size) and ...));

  protected:
    template<typename T, typename ColumnCoefficients = OutputCoefficients>
    static constexpr bool is_valid_input_matrix_v()
    {
      static_assert(is_typed_matrix_v<T> or is_typed_matrix_base_v<T>);
      if constexpr(is_typed_matrix_v<T>)
        return
          is_equivalent_v<typename MatrixTraits<T>::RowCoefficients, OutputCoefficients> and
          is_equivalent_v<typename MatrixTraits<T>::ColumnCoefficients, ColumnCoefficients>;
      else
        return
          MatrixTraits<T>::dimension == OutputCoefficients::size and
          MatrixTraits<T>::columns == ColumnCoefficients::size;
    }

    using TransformationMatricesTuple = std::tuple<TypedMatrix<OutputCoefficients, InputCoefficients, TransformationMatrix>,
      TypedMatrix<OutputCoefficients, OutputCoefficients, PerturbationTransformationMatrices>...>;
    const TransformationMatricesTuple transformation_matrices;

    using FunctionImpl = internal::FunctionImpl<TransformationMatricesTuple>;
    using JacobianImpl = internal::JacobianImpl<TransformationMatricesTuple>;
    using Base = Transformation<InputCoefficients, OutputCoefficients, FunctionImpl, JacobianImpl>;

  public:
    LinearTransformation(const TransformationMatrix& mat, const PerturbationTransformationMatrices& ... p_mats)
      : Base(FunctionImpl(transformation_matrices), JacobianImpl(transformation_matrices)),
        transformation_matrices(mat, p_mats...) {}

    template<typename T, typename ... Ps,
      std::enable_if_t<(is_typed_matrix_v<T> or is_typed_matrix_base_v<T>) and
        ((is_typed_matrix_v<Ps> or is_typed_matrix_base_v<Ps>) and ...), int> = 0>
    LinearTransformation(const T& mat, const Ps& ... p_mats)
      : Base(FunctionImpl(transformation_matrices), JacobianImpl(transformation_matrices)),
        transformation_matrices(mat, p_mats...)
    {
      static_assert(is_valid_input_matrix_v<T, InputCoefficients>);
      static_assert((is_valid_input_matrix_v<Ps, OutputCoefficients> and ...));
    }
  };


  /**
   * Deduction guides
   */

  template<typename T, typename ... Ps,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<T>,
      std::disjunction<is_typed_matrix<Ps>, is_typed_matrix_base<Ps>>...>, int> = 0>
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    typename MatrixTraits<T>::ColumnCoefficients,
    typename MatrixTraits<T>::RowCoefficients,
    typename MatrixTraits<T>::BaseMatrix,
    std::conditional_t<is_typed_matrix_v<Ps>, typename MatrixTraits<Ps>::BaseMatrix, std::decay_t<Ps>>...>;

  template<typename T, typename ... Ps,
    std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Ps>...>, int> = 0>
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::dimension>,
    std::decay_t<T>,
    std::decay_t<Ps>...>;

}


#endif //OPENKALMAN_LINEARTRANSFORMATION_H
