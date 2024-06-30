/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref reduce function.
 */

#ifndef OPENKALMAN_REDUCE_HPP
#define OPENKALMAN_REDUCE_HPP

namespace OpenKalman
{
  // -------- //
  //  reduce  //
  // -------- //

  namespace detail
  {
    template<std::size_t...indices, typename T, std::size_t...Is>
    constexpr auto count_reduced_dimensions(const T& t, std::index_sequence<Is...>)
    {
      if constexpr ((dynamic_dimension<T, indices> or ...))
      {
        return ([](const T& t){
          constexpr auto I = Is;
          if constexpr (((I == indices) or ...)) return get_index_dimension_of<I>(t);
          else return 1;
        }(t) * ... * 1);
      }
      else
      {
        constexpr auto dim = ([]{
          constexpr auto I = Is;
          if constexpr (((I == indices) or ...)) return index_dimension_of_v<T, I>;
          else return 1;
        }() * ... * 1);
        return std::integral_constant<std::size_t, dim>{};
      }
    }


#ifdef __cpp_concepts
    template<index_value Dim, typename BinaryOperation, typename Constant>
#else
    template<typename Dim, typename BinaryOperation, typename Constant, std::enable_if_t<index_value<Dim>, int> = 0>
#endif
    constexpr auto scalar_reduce_operation(const Dim& dim, const BinaryOperation& op, const Constant& c)
    {
      if constexpr (internal::is_plus<BinaryOperation>::value) return c * dim;
      else if constexpr (internal::is_multiplies<BinaryOperation>::value) return internal::constexpr_pow(c, dim);
      else
      {
        if constexpr (dynamic_index_value<Dim>)
        {
          if (get_scalar_constant_value(dim) <= 1) return get_scalar_constant_value(c);
          else return op(scalar_reduce_operation(get_scalar_constant_value(dim) - 1, op, c), get_scalar_constant_value(c));
        }
        else if constexpr (Dim::value <= 1) return c;
        else
        {
          auto dim_m1 = std::integral_constant<std::size_t, Dim::value - 1>{};
          return values::scalar_constant_operation {op, scalar_reduce_operation(dim_m1, op, c), c};
        }
      }
    }

  } // namespace detail


  /**
   * \brief Perform a partial reduction based on an associative binary function, across one or more indices.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indices to be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, indexible Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (internal::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
  constexpr indexible decltype(auto)
#else
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (internal::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
  constexpr decltype(auto)
#endif
  reduce(BinaryFunction&& b, Arg&& arg)
  {
    // \todo Check if Arg is already reduced or partially reduced.
    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 1>>, vector_space_descriptor_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(std::forward<BinaryFunction>(b), to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(reduce<index, indices...>(std::forward<BinaryFunction>(b), nested_object(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      auto m = reduce<index, indices...>(std::forward<BinaryFunction>(b), nested_object(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 0>>, vector_space_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<vector_space_descriptor_of_t<Arg, 1>>, vector_space_descriptor_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(std::forward<BinaryFunction>(b), nested_object(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (dimension_size_of_index_is<Arg, index, 1>)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }
    else if constexpr (zero<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
    {
      constexpr std::make_index_sequence<index_count_v<Arg>> seq;
      return internal::make_constant_matrix_reduction<index, indices...>(
        values::ScalarConstant<scalar_type_of_t<Arg>, 0>{}, std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr std::make_index_sequence<index_count_v<Arg>> seq;
      auto dim = detail::count_reduced_dimensions<index, indices...>(arg, seq);
      auto c = detail::scalar_reduce_operation(dim, b, constant_coefficient{arg});
      return internal::make_constant_matrix_reduction<index, indices...>(std::move(c), std::forward<Arg>(arg), seq);
    }
    // \todo Add this after updating DiagonalMatrix to include diagonals of any rank
    //else if constexpr (constant_diagonal_matrix<Arg>)
    //{
    //}
    else
    {
      using Lib = interface::library_interface<std::decay_t<Arg>>;
      auto red = Lib::template reduce<index, indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      if constexpr (scalar_type<decltype(red)>) return make_constant<Arg>(std::move(red), Dimensions<1>{});
      else return red;
    }
  }


  namespace detail
  {
    template<typename BinaryFunction, typename Arg, std::size_t...indices>
    constexpr scalar_type_of_t<Arg>
    reduce_all_indices(const BinaryFunction& b, const Arg& arg, std::index_sequence<indices...>)
    {
      if constexpr (zero<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
      {
        return 0;
      }
      else if constexpr (one_dimensional<Arg>)
      {
        return get_component(arg, static_cast<decltype(indices)>(0)...);
      }
      else if constexpr (constant_matrix<Arg>)
      {
        constexpr auto seq = std::make_index_sequence<index_count_v<Arg>> {};
        auto dim = count_reduced_dimensions<indices...>(arg, seq);
        return scalar_reduce_operation(dim, b, constant_coefficient {arg});
      }
      else
      {
        auto red = interface::library_interface<Arg>::template reduce<indices...>(b, arg);
        using Red = decltype(red);

        static_assert(scalar_type<Red> or one_dimensional<Red, Qualification::depends_on_dynamic_shape>,
          "Incorrect library interface for total 'reduce' on all indices: must return a scalar or one-by-one matrix.");

        if constexpr (scalar_type<Red>)
          return red;
        else
          return get_component(red, static_cast<decltype(indices)>(0)...);
      }
    }

  } // namespace detail

  /**
   * \overload
   * \brief Perform a complete reduction based on an associative binary function, and return a scalar.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A scalar representing a complete reduction.
   */
#ifdef __cpp_concepts
  template<typename BinaryFunction, indexible Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (internal::has_uniform_reduction_indices<Arg>(std::make_index_sequence<index_count_v<Arg>> {}))
#else
  template<typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (internal::has_uniform_reduction_indices<Arg>(std::make_index_sequence<index_count<Arg>::value> {})), int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  reduce(const BinaryFunction& b, const Arg& arg)
  {
    return detail::reduce_all_indices(b, arg, std::make_index_sequence<index_count_v<Arg>> {});
  }


} // namespace OpenKalman

#endif //OPENKALMAN_REDUCE_HPP
