/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Element-wise n-ary operation function.
 */

#ifndef OPENKALMAN_N_ARY_OPERATION_HPP
#define OPENKALMAN_N_ARY_OPERATION_HPP

#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/dynamic_dimension.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/traits/layout_of.hpp"

namespace OpenKalman
{
  // ----------------- //
  //  n_ary_operation  //
  // ----------------- //

  namespace detail
  {

    // Check that dimensions of arguments Args are compatible with \ref patterns::pattern Ds.
    template<std::size_t...ixs, typename DTup, typename...Args>
    constexpr void check_n_ary_dims(std::index_sequence<ixs...>, const DTup& d_tup, const Args&...args)
    {
      return ([](const DTup& d_tup, const Args&...args){
        constexpr auto ix = ixs;
        return ([](const DTup& d_tup, const auto& arg){
          using Arg = decltype(arg);
          if constexpr (dynamic_dimension<Arg, ix> or patterns::dynamic_pattern<collections::collection_element_t<ix, DTup>>)
          {
            auto arg_d = get_pattern_collection<ix>(arg);
            auto tup_d = std::get<ix>(d_tup);
            auto dim_arg_d = get_dimension(arg_d);
            auto dim_tup_d = get_dimension(tup_d);
            if (not (arg_d == tup_d) and not patterns::is_uniform_pattern_component_of(arg_d, tup_d))
              throw std::logic_error {"In an argument to n_ary_operation, the dimension of index " +
                std::to_string(ix) + " is " + std::to_string(dim_arg_d) + ", but should be 1 " +
                (dim_tup_d == 1 ? "" : "or " + std::to_string(dim_tup_d)) +
                "(the dimension of Ds number " + std::to_string(ix)};
          }
          else
          {
            using D_Arg = std::decay_t<decltype(get_index_pattern(arg, std::integral_constant<std::size_t, ix>{}))>;
            using D = collections::collection_element_t<ix, DTup>;
            static_assert(patterns::compares_with<D_Arg, D> or equivalent_to_uniform_pattern_component_of<D_Arg, D> or
              (ix >= index_count_v<Arg> and patterns::uniform_pattern<D>),
              "In argument to n_ary_operation, the dimension of each index must be either 1 or that of Ds.");
          }
        }(d_tup, args),...);
      }(d_tup, args...),...);
    }


    template<typename Op, typename...Args, std::size_t...I>
    constexpr bool is_invocable_with_indices(std::index_sequence<I...>)
    {
      return std::is_invocable_v<Op, Args..., decltype(I)...>;
    }


    template<typename Op, std::size_t...I, typename...Args>
    constexpr decltype(auto) n_ary_invoke_op(const Op& op, std::index_sequence<I...> seq, Args&&...args)
    {
      if constexpr (is_invocable_with_indices<const Op&, Args&&...>(seq))
        return op(std::forward<Args>(args)..., static_cast<decltype(I)>(0)...);
      else
        return op(std::forward<Args>(args)...);
    }


#ifdef __cpp_concepts
    template<typename Operation, std::size_t indices, typename...Args>
#else
    template<typename Operation, std::size_t indices, typename = void, typename...Args>
#endif
    struct n_ary_operator_traits_impl {};


#ifdef __cpp_concepts
    template<typename Op, std::size_t indices, typename...Args>
    requires (is_invocable_with_indices<Op, Args...>(std::make_index_sequence<indices> {})) or
      std::is_invocable_v<Op, Args...>
    struct n_ary_operator_traits_impl<Op, indices, Args...>
#else
    template<typename Op, std::size_t indices, typename...Args>
    struct n_ary_operator_traits_impl<Op, indices, std::enable_if_t<
      is_invocable_with_indices<Op, Args...>(std::make_index_sequence<indices> {}) or
      std::is_invocable_v<Op, Args...>>, Args...>
#endif
    {
      using type = decltype(n_ary_invoke_op(std::declval<Op>(), std::make_index_sequence<indices> {}, std::declval<Args>()...));
    };


    template<typename Op, std::size_t indices, typename...Args>
    struct n_ary_operator_traits
#ifdef __cpp_concepts
      : n_ary_operator_traits_impl<Op, indices, Args...> {};
#else
      : n_ary_operator_traits_impl<Op, indices, void, Args...> {};
#endif


#ifndef __cpp_concepts
    template<typename Op, std::size_t Indices, typename = void, typename...Args>
    struct n_ary_operator_impl : std::false_type {};

    template<typename Op, std::size_t Indices, typename...Args>
    struct n_ary_operator_impl<Op, Indices, std::enable_if_t<
      std::is_invocable<Op, typename std::add_lvalue_reference<typename element_type_of<Args>::type>::type...>::value or
      is_invocable_with_indices<Op, typename std::add_lvalue_reference<typename element_type_of<Args>::type>::type...>(
        std::make_index_sequence<Indices> {})>, Args...>
    : std::true_type {};
#endif


    template<typename Op, std::size_t Indices, typename...Args>
#ifdef __cpp_concepts
    concept n_ary_operator = std::is_invocable_v<Op, std::add_lvalue_reference_t<element_type_of_t<Args>>...> or
      is_invocable_with_indices<Op, std::add_lvalue_reference_t<element_type_of_t<Args>>...>(std::make_index_sequence<Indices> {});
#else
    constexpr bool n_ary_operator = n_ary_operator_impl<Op, Indices, void, Args...>::value;
#endif


    template<typename Arg, std::size_t...I, typename...J>
    inline auto n_ary_operation_get_component_impl(Arg&& arg, std::index_sequence<I...>, J...j)
    {
      if constexpr (sizeof...(I) == sizeof...(J))
        return access(std::forward<Arg>(arg), (j < get_index_extent<I>(arg) ? j : 0)...);
      else
        return access(std::forward<Arg>(arg), [](auto dim, const auto& j_tup){
          auto j = std::get<I>(j_tup);
          if (j < dim) return j;
          else return 0_uz;
        }(get_index_extent<I>(arg), std::tuple {j...})...);
    }


    template<typename Op, typename ArgsTup, std::size_t...ArgI, typename...J>
    inline auto n_ary_operation_get_component(const Op& op, ArgsTup&& args_tup, std::index_sequence<ArgI...>, J...j)
    {
      if constexpr (std::is_invocable_v<const Op&, element_type_of_t<collections::collection_element_t<ArgI, ArgsTup>>..., J...>)
        return op(n_ary_operation_get_component_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<index_count_v<collections::collection_element_t<ArgI, ArgsTup>>> {},
          j...)..., j...);
      else
        return op(n_ary_operation_get_component_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<index_count_v<collections::collection_element_t<ArgI, ArgsTup>>> {},
          j...)...);
    }


    template<typename M, typename Op, typename ArgsTup, typename...J>
    inline void n_ary_operation_iterate(M& m, const Op& op, ArgsTup&& args_tup, std::index_sequence<>, J...j)
    {
      std::make_index_sequence<collections::size_of_v<ArgsTup>> seq;
      set_component(m, n_ary_operation_get_component(op, std::forward<ArgsTup>(args_tup), seq, j...), j...);
    }


    template<typename M, typename Op, typename ArgsTup, std::size_t I, std::size_t...Is, typename...J>
    inline void n_ary_operation_iterate(M& m, const Op& op, ArgsTup&& args_tup, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_extent<I>(m); i++)
        n_ary_operation_iterate(m, op, std::forward<ArgsTup>(args_tup), std::index_sequence<Is...> {}, j..., i);
    }


    /// Replicate an object, if necessary, to expand any 1D indices to fill a particular shape.
    template<typename...Ds, typename Arg, std::size_t...Ix_Ds>
    static decltype(auto)
    replicate_arg(const std::tuple<Ds...>& d_tup, Arg&& arg, std::index_sequence<Ix_Ds...>)
    {
      OpenKalman::broadcast(std::forward<Arg>(arg),
        values::operation(
          std::divides<element_type_of_t<Arg>>{},
          get_dimension(std::get<Ix_Ds>(d_tup)),
          get_index_extent<Ix_Ds>(arg))...);
    }


    template<typename PatternMatrix, typename...Ds, typename Op, typename...Args>
    static constexpr auto
    n_ary_operation_impl(const std::tuple<Ds...>& d_tup, Op&& op, Args&&...args)
    {
      constexpr std::index_sequence_for<Ds...> seq;

      // constant_object:
      if constexpr (sizeof...(Args) > 0 and (constant_object<Args> and ...) and not is_invocable_with_indices<Op, element_type_of_t<Args>...>(seq))
      {
        values::operation c {op, constant_value {std::forward<Args>(args)}...};
        return std::apply(
          [](auto&&...as){ return make_constant<PatternMatrix>(std::forward<decltype(as)>(as)...); },
          std::tuple_cat(std::tuple{std::move(c)}, d_tup));
      }
      // Library handles n-ary operation.
      else if constexpr (is_invocable_with_indices<Op, std::add_lvalue_reference_t<element_type_of_t<Args>>...>(seq) and
        interface::n_ary_operation_defined_for<PatternMatrix, const std::tuple<Ds...>&, Op&&, Args&&...>)
      {
        using Interface = interface::library_interface<stdex::remove_cvref_t<PatternMatrix>>;
        auto ret {Interface::n_ary_operation(d_tup, std::forward<Op>(op), replicate_arg(d_tup, std::forward<Args>(args), seq)...)};
        return std::apply([](auto&& a, auto&&...vs){
          return attach_pattern(std::forward<decltype(a)>(a), std::forward<decltype(vs)>(vs)...);
          }, std::tuple_cat(std::forward_as_tuple(std::move(ret)), d_tup));
      }
      // \todo add case where interface needs arguments to be wrapped in a library adapter
      else // Catch-all: library does not provide for this n-ary operation.
      {
        using Scalar = std::decay_t<typename n_ary_operator_traits<Op, sizeof...(Ds),
          std::add_lvalue_reference_t<element_type_of_t<Args>>...>::type>;

        if constexpr (((patterns::dimension_of_v<Ds> == 1) and ...))
        {
          // one-by-one matrix
          auto e = op(access(std::forward<Args>(args))...);
          return make_dense_object_from<PatternMatrix, layout_of_t<PatternMatrix>, Scalar>(d_tup, e);
        }
        else
        {
          auto m = std::apply([](auto&&...ds){
            return make_dense_object<PatternMatrix, layout_of_t<PatternMatrix>, Scalar>(std::forward<decltype(ds)>(ds)...);
          }, d_tup);
          n_ary_operation_iterate(m, op, std::forward_as_tuple(std::forward<Args>(args)...), seq);
          return m;
        }
      }
    }

  }


  /**
   * \brief Perform a component-wise n-ary operation, using broadcasting to match the size of a pattern matrix.
   * \details This overload is for unary, binary, and higher n-ary operations. Examples:
   * - Unary operation, no broadcasting:
   *   \code
   *     auto ds32 = std::tuple {Dimensions<3>{}, Dimensions<2>{}};
   *     auto op1 = [](auto arg){return 3 * arg;};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto m32 = make_dense_object_from<M>(ds32, 1, 2, 3, 4, 5, 6);
   *     std::cout << n_ary_operation(ds32, op1, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     3, 6,
   *     9, 12,
   *     15, 18
   *   \endcode
   * - Unary operation, broadcasting:
   *   \code
   *     auto ds31 = std::tuple {Dimensions<3>{}, Dimensions<1>{}};
   *     auto m31 = make_dense_object_from<M>(ds31, 1, 2, 3);
   *     std::cout << n_ary_operation(ds32, op1, m31) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     3, 3,
   *     6, 6,
   *     9, 9
   *   \endcode
   * - Binary operation, no broadcasting:
   *   \code
   *     auto op2 = [](auto arg1, auto arg2){return 3 * arg1 + arg2;};
   *     std::cout << n_ary_operation(ds32, op2, m32, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 10,
   *     15, 20,
   *     25, 30
   *   \endcode
   * - Binary operation, broadcasting:
   *   \code
   *     std::cout << n_ary_operation(ds32, op2, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 7,
   *     12, 14,
   *     19, 21
   *   \endcode
   * - Binary operation, broadcasting, with indices:
   *   \code
   *     auto op2b = [](auto arg1, auto arg2, std::size_t row, std::size_t col){return 3 * arg1 + arg2 + row + col;};
   *     std::cout << n_ary_operation(ds32, op2b, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 8,
   *     13, 16,
   *     21, 24
   *   \endcode
   * \tparam Ds \ref patterns::pattern objects defining the size of the result.
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices indicating the location
   * within the result. The operation must return a scalar value.
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<patterns::pattern...Ds, typename Operation, indexible...Args> requires (sizeof...(Args) > 0) and
    detail::n_ary_operator<Operation, sizeof...(Ds), Args...> and (... and (patterns::dimension_of_v<Ds> != 0))
  constexpr compares_with_pattern_collection<std::tuple<Ds...>> auto
#else
  template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<
    (patterns::pattern<Ds> and ...) and (indexible<Args> and ...) and (sizeof...(Args) > 0) and
    detail::n_ary_operator<Operation, sizeof...(Ds), Args...> and (... and (patterns::dimension_of_v<Ds> != 0)), int> = 0>
  constexpr auto
#endif
  n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& operation, Args&&...args)
  {
    detail::check_n_ary_dims(std::index_sequence_for<Ds...> {}, d_tup, args...);
    using Arg0 = std::decay_t<collections::collection_element_t<0, std::tuple<Args...>>>; // \todo Pick the first appropriate pattern matrix, even if not the first one.
    return detail::n_ary_operation_impl<Arg0>(d_tup, std::forward<Operation>(operation), std::forward<Args>(args)...);
  }


  namespace detail
  {
    template<std::size_t ix, typename Arg, typename...Args>
    constexpr auto find_max_dim(const Arg& arg, const Args&...args)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        auto ret = get_pattern_collection<ix>(arg);
        if constexpr (patterns::dynamic_pattern<decltype(ret)>)
        {
          if (get_dimension(ret) == 0) throw std::invalid_argument {"A dimension of an arguments "
            "to n_ary_operation is zero for at least index " + std::to_string(ix) + "."};
        }
        else static_assert(index_dimension_of_v<Arg, ix> != 0, "Arguments to n_ary_operation cannot have zero dimensions");
        return ret;
      }
      else
      {
        auto max_d = find_max_dim<ix>(args...);
        using Arg_D = std::decay_t<decltype(get_index_pattern(arg, std::integral_constant<std::size_t, ix>{}))>;
        using Max_D = decltype(max_d);

        if constexpr (fixed_pattern<Arg_D> and fixed_pattern<Max_D>)
        {
          constexpr auto dim_arg_d = patterns::dimension_of_v<Arg_D>;
          if constexpr (compares_with<Arg_D, Max_D>or (dim_arg_d == 1 and equivalent_to_uniform_pattern_component_of<Arg_D, Max_D>))
          {
            return max_d;
          }
          else
          {
            constexpr auto dim_max_d = patterns::dimension_of_v<Max_D>;
            static_assert(dim_max_d != 1 or not equivalent_to_uniform_pattern_component_of<Max_D, Arg_D>,
              "The dimension of arguments to n_ary_operation are not compatible with each other for at least one index.");
            return get_pattern_collection<ix>(arg);
          }
        }
        else if constexpr (patterns::euclidean_pattern<Arg_D> and patterns::euclidean_pattern<Max_D>)
        {
          if constexpr (fixed_pattern<Arg_D>)
          {
            constexpr std::size_t a = patterns::dimension_of_v<Arg_D>;
            std::size_t m = get_dimension(max_d);
            if (a != m and a != 1 and m != 1) throw std::invalid_argument {"The dimension of arguments to n_ary_operation "
                "are not compatible with each other for at least index " + std::to_string(ix) + "."};

            if constexpr (a == 1) return max_d;
            else return get_pattern_collection<ix>(arg);
          }
          else if constexpr (fixed_pattern<Max_D>)
          {
            auto arg_d = get_pattern_collection<ix>(arg);
            std::size_t a = get_dimension(arg_d);
            constexpr std::size_t m = patterns::dimension_of_v<Max_D>;
            if (a != m and a != 1 and m != 1) throw std::invalid_argument {"The dimension of arguments to n_ary_operation "
                "are not compatible with each other for at least index " + std::to_string(ix) + "."};

            if constexpr (m == 1) return arg_d;
            else return max_d;
          }
          else
          {
            std::size_t a = get_index_extent<ix>(arg);
            std::size_t m = get_dimension(max_d);
            if (a == m or a == 1) return m;
            else if (m == 1 and m <= a) return a;
            else throw std::invalid_argument {"The dimension of arguments to n_ary_operation are not compatible with "
              "each other for at least index " + std::to_string(ix) + "."};
          }
        }
        else
        {
          auto arg_d = get_pattern_collection<ix>(arg);
          //using Scalar = element_type_of_t<Arg>;
          if (patterns::is_uniform_pattern_component_of(arg_d, max_d))
          {
            //if constexpr (internal::is_DynamicDescriptor<Max_D>::value) return DynamicDescriptor {max_d};
            //else return DynamicDescriptor<Scalar> {max_d};
            return max_d;
          }
          else if (patterns::is_uniform_pattern_component_of(max_d, arg_d))
          {
            //if constexpr (internal::is_DynamicDescriptor<Max_D>::value) return DynamicDescriptor {arg_d};
            //else return DynamicDescriptor<Scalar> {arg_d};
            return arg_d;
          }
          else throw std::invalid_argument {"The dimension of arguments to n_ary_operation are not compatible with "
            "each other for at least index " + std::to_string(ix) + "."};
        }
      }
    }


    template<std::size_t...ixs, typename...Args>
    constexpr auto find_max_dims(std::index_sequence<ixs...>, const Args&...args)
    {
      return std::tuple {find_max_dim<ixs>(args...)...};
    }

  }


  /**
   * \overload
   * \brief Perform a component-wise n-ary operation, using broadcasting if necessary to make the arguments the same size.
   * \details Each of the arguments may be expanded by broadcasting. The result will derive each dimension from the
   * largest corresponding dimension among the arguments.
   * Examples:
   * - Binary operation, broadcasting:
   *   \code
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto op2a = [](auto arg1, auto arg2){return 3 * arg1 + arg2;};
   *     auto m31 = make_dense_object_from<M>(ds31, 1, 2, 3);
   *     auto m32 = make_dense_object_from<M>(ds32, 1, 2, 3, 4, 5, 6);
   *     std::cout << n_ary_operation(op2a, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 7,
   *     12, 14,
   *     19, 21
   *   \endcode
   * - Binary operation, broadcasting, with indices:
   *   \code
   *     auto op2b = [](auto arg1, auto arg2, std::size_t row, std::size_t col){return 3 * arg1 + arg2 + row + col;};
   *     std::cout << n_ary_operation(op2b, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 8,
   *     13, 16,
   *     21, 24
   *   \endcode
   * - Unary operation, with indices:
   *   \code
   *     auto op1a = [](auto& arg, std::size_t row, std::size_t col){return arg + row + col;};
   *     std::cout << n_ary_operation(op1a, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices. The operation must
   * return a scalar value.
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<typename Operation, indexible...Args> requires (sizeof...(Args) > 0) and
    detail::n_ary_operator<Operation, std::max({index_count_v<Args>...}), Args...>
#else
  template<typename Operation, typename...Args, std::enable_if_t<(indexible<Args> and ...) and
    (sizeof...(Args) > 0) and detail::n_ary_operator<Operation, std::max({index_count<Args>::value...}), Args...>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(Operation&& operation, Args&&...args)
  {
    auto d_tup = detail::find_max_dims(std::make_index_sequence<std::max({index_count_v<Args>...})> {}, args...);
    using Arg0 = std::decay_t<collections::collection_element_t<0, std::tuple<Args...>>>;
    return detail::n_ary_operation_impl<Arg0>(std::move(d_tup), std::forward<Operation>(operation), std::forward<Args>(args)...);
  }


  namespace detail
  {
    template<typename M, typename Operation, typename Vs_tuple, typename Index_seq, typename K_seq, typename...Is>
    void nullary_set_components(M& m, const Operation& op, const Vs_tuple&, Index_seq, K_seq, Is...is)
    {
      constexpr auto seq = std::make_index_sequence<sizeof...(Is)> {};
      if constexpr (detail::is_invocable_with_indices<Operation>(seq))
        set_component(m, op(is...), is...); //< Operation takes a full set of indices.
      else
        set_component(m, op(), is...); //< Operation takes no indices.
    }


    template<std::size_t DsIndex, std::size_t...DsIndices, typename M, typename Operation,
      typename Vs_tuple, std::size_t...indices, std::size_t...Ks, typename...Is>
    void nullary_set_components(M& m, const Operation& op, const Vs_tuple& ds_tup,
      std::index_sequence<indices...> index_seq, std::index_sequence<Ks...> k_seq, Is...is)
    {
      if constexpr (((DsIndex == indices) or ...))
      {
        constexpr std::integral_constant<size_t, ((DsIndex == indices ? Ks : 0) + ...)> i;
        nullary_set_components<DsIndices...>(m, op, ds_tup, index_seq, k_seq, is..., i);
      }
      else
      {
        // Iterate through the dimensions of the current DsIndex and add set elements for each dimension iteratively.
        for (std::size_t i = 0; i < get_dimension(std::get<DsIndex>(ds_tup)); ++i)
        {
          nullary_set_components<DsIndices...>(m, op, ds_tup, index_seq, k_seq, is..., i);
        }
      }
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, typename M, typename Operations_tuple,
      typename Vs_tuple, typename UniqueIndicesSeq, std::size_t...AllDsIndices, typename K_seq>
    void nullary_iterate(M& m, const Operations_tuple& op_tup, const Vs_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, std::index_sequence<AllDsIndices...>, K_seq k_seq)
    {
      nullary_set_components<AllDsIndices...>(m, std::get<CurrentOpIndex>(op_tup), ds_tup, unique_indices_seq, k_seq);
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, std::size_t index, std::size_t...indices,
      typename M, typename Operations_tuple, typename Vs_tuple, typename UniqueIndicesSeq, typename AllDsSeq,
      std::size_t...Ks, std::size_t...Js, typename...J_seqs>
    void nullary_iterate(M& m, const Operations_tuple& op_tup, const Vs_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, AllDsSeq all_ds_seq, std::index_sequence<Ks...>, std::index_sequence<Js...>,
      J_seqs...j_seqs)
    {
      constexpr std::size_t new_factor = factor / patterns::dimension_of_v<collections::collection_element_t<index, Vs_tuple>>;

      (nullary_iterate<CurrentOpIndex + new_factor * Js, new_factor, indices...>(
        m, op_tup, ds_tup, unique_indices_seq, all_ds_seq, std::index_sequence<Ks..., Js>{}, j_seqs...),...);
    }

  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation with potentially multiple operations for different blocks.
   * \details Examples:
   * - One operation for the entire matrix
   *   \code
   *     auto ds23 = std::tuple {Dimensions<2>{}, Dimensions<3>{}};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     std::cout << n_ary_operation<M>(ds23, [](auto arg){return 7;}) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     7, 7, 7,
   *     7, 7, 7
   *   \endcode
   * - One operation for each element
   *   \code
   *     std::cout << n_ary_operation<M, 0, 1>(ds23, []{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
   *   \endcode
   *   Output:
   *   \code
   *     4, 5, 6,
   *     7, 8, 9
   *   \endcode
   * - One operation for each row
   *   \code
   *     auto ds23a = std::tuple {Dimensions<2>{}, Dimensions{3}};
   *     std::cout << n_ary_operation<M, 0>(ds23a, []{return 5;}, []{return 6;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 5, 5,
   *     6, 6, 6
   *   \endcode
   * - One operation for each column
   *   \code
   *     auto ds23b = std::tuple {Dimensions{2}, Dimensions<3>{}};
   *     std::cout << n_ary_operation<M, 1>(ds23b, []{return 5;}, []{return 6;}, []{return 7;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 6, 7,
   *     5, 6, 7
   *   \endcode
   * - One operation for each column, with indices
   *   \code
   *     auto ds23b = std::tuple {Dimensions{2}, Dimensions<3>{}};
   *     auto op1 = [](std::size_t r, std::size_t c){ return 5 + r + c; };
   *     auto op2 = [](std::size_t r, std::size_t c){ return 6 + r + c; };
   *     auto op3 = [](std::size_t r, std::size_t c){ return 7 + r + c; };
   *     std::cout << n_ary_operation<M, 1>(ds23b, []{return 5;}, []{return 6;}, []{return 7;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 7, 9,
   *     6, 8, 10
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its purpose is to indicate the
   * library from which to create a resulting matrix, and its dimensions need not match the specified dimensions Ds
   * \tparam indices The indices, if any, along which there will be a different operator for each element along that index.
   * \tparam Ds \ref patterns::pattern objects for each index of the result. \ref patterns::pattern objects corresponding to indices must be
   * of a \ref fixed_pattern type.
   * \tparam Operations The nullary operations. The number of operations must equal the product of each dimension Ds
   * corresponding to indices. The order of the operations depends on the order of indices, with the left-most index
   * being the most major, and the right-most index being the most minor. For example, if indices
   * are {0, 1} for a matrix result, the operations must be in row-major order. If the indices are {1, 0}, the
   * operations must be in column-major order.
   * Each operation may be invocable with no arguments or invocable with as many indices
   * as there are \ref patterns::pattern Ds. (If the index corresponds to one of designaged <code>indices</code>, then the
   * operation will be called with <code>std::integral_constant<std::size_t, _index_>{}</code> for that index,
   * instead of <code>std::size_t</code>.))
   * \return A matrix or array in which each component is the result of calling Operation and which has
   * dimensions corresponding to Ds
   */
  #ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, patterns::pattern...Ds, typename...Operations>
  requires ((fixed_pattern<collections::collection_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == (1 * ... * patterns::dimension_of_v<collections::collection_element_t<indices, std::tuple<Ds...>>>)) and
    (detail::n_ary_operator<Operations, sizeof...(Ds)> and ...)
  #else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (patterns::pattern<Ds> and ...) and
    ((fixed_pattern<collections::collection_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == (1 * ... * patterns::dimension_of<collections::collection_element_t<indices, std::tuple<Ds...>>>::value)) and
    (detail::n_ary_operator<Operations, sizeof...(Ds)> and ...), int> = 0>
  #endif
  constexpr auto
  n_ary_operation(const std::tuple<Ds...>& d_tup, const Operations&...operations)
  {
    using Scalar = std::common_type_t<std::decay_t<typename detail::n_ary_operator_traits<Operations, sizeof...(Ds)>::type>...>;

    // One operation for all elements combined:
    if constexpr (sizeof...(Operations) == 1)
    {
      return detail::n_ary_operation_impl<std::decay_t<PatternMatrix>>(d_tup, operations...);
    }
    // One operation for each element, and the operations are not invocable with indices:
    else if constexpr (((not dynamic_pattern<Ds>) and ...) and
      sizeof...(operations) == (patterns::dimension_of_v<Ds> * ...) and
      not (detail::is_invocable_with_indices<const Operations&>(std::make_index_sequence<sizeof...(Ds)> {}) or ...))
    {
      return make_dense_object_from<PatternMatrix, layout_of_t<PatternMatrix>, Scalar>(d_tup, operations()...);
    }
    // All other cases:
    else
    {
      auto m = make_dense_object<PatternMatrix, layout_of_t<PatternMatrix>, Scalar>(d_tup);
      detail::nullary_iterate<0, sizeof...(Operations), indices...>(
        m,
        std::forward_as_tuple(operations...),
        d_tup,
        std::index_sequence<indices...> {},
        std::index_sequence_for<Ds...> {},
        std::index_sequence<> {},
        std::make_index_sequence<patterns::dimension_of_v<collections::collection_element_t<indices, std::tuple<Ds...>>>> {}...);
      return m;
    }
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation, deriving the resulting size from a pattern matrix.
     * \details
     * - One operation for the entire matrix
     *   \code
     *     auto M = Eigen::Matrix<double, 2, 3>
     *     std::cout << n_ary_operation<M>([](auto arg){return 7;}) << std::endl;
     *   \endcode
     *   Output:
     *   \code
     *     7, 7, 7,
     *     7, 7, 7
     *   \endcode
     * - One operation for each element
     *   \code
     *     std::cout << n_ary_operation<M, 0, 1>([]{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
     *   \endcode
     *   Output:
     *   \code
     *     4, 5, 6,
     *     7, 8, 9
     *   \endcode
     * - One operation for each row
     *   \code
     *     std::cout << n_ary_operation<M, 0>([]{return 5;}, []{return 6;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 5, 5,
     *     6, 6, 6
     *   \endcode
     * - One operation for each column
     *   \code
     *     std::cout << n_ary_operation<M, 1>([]{return 5;}, []{return 6;}, []{return 7;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 6, 7,
     *     5, 6, 7
     *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its purpose is to indicate the
   * library from which to create a resulting matrix, and its dimensions need not match the specified dimensions Ds.
   * Dimensions of PatternMatrix corresponding to indices must be of a \ref fixed_pattern type.
   * \tparam indices The indices, if any, along which there will be a different operator for each element along that index.
   * \tparam Operations The nullary operations. The number of operations must equal the product of each dimension Ds
   * corresponding to indices.
   * \return A matrix or array in which each component is the result of calling Operation and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, patterns::pattern...Ds, typename...Operations>
  requires ((not dynamic_dimension<PatternMatrix, indices>) and ...) and
    (sizeof...(Operations) == (1 * ... * index_dimension_of_v<PatternMatrix, indices>))
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (patterns::pattern<Ds> and ...) and
    ((not dynamic_dimension<PatternMatrix, indices>) and ...) and
    (sizeof...(Operations) == (1 * ... * index_dimension_of<PatternMatrix, indices>::value)), int> = 0>
#endif
  constexpr auto
  n_ary_operation(const Operations&...operations)
  {
    auto d_tup = get_pattern_collection<PatternMatrix>();
    return n_ary_operation<PatternMatrix, indices...>(d_tup, operations...);
  }


  // -------------------------- //
  //  n_ary_operation_in_place  //
  // -------------------------- //

  namespace detail
  {
    template<typename Operation, typename Elem, typename...J>
    inline void do_elem_operation_in_place_impl(const Operation& operation, Elem& elem, J...j)
    {
      if constexpr (std::is_invocable_v<const Operation&, Elem&, J...>)
        operation(elem, j...);
      else
        operation(elem);
    }


    template<typename Operation, typename Arg, typename...J>
    inline void do_elem_operation_in_place(const Operation& operation, Arg& arg, J...j)
    {
      auto&& elem = access(arg, j...);
      if constexpr (std::is_assignable_v<decltype((elem)), std::decay_t<decltype(elem)>>)
      {
        do_elem_operation_in_place_impl(operation, elem, j...);
      }
      else
      {
        auto e {std::forward<decltype(elem)>(elem)}; // copy elem
        static_assert(std::is_assignable_v<decltype((e)), std::decay_t<decltype(elem)>>);
        do_elem_operation_in_place_impl(operation, e, j...);
        set_component(arg, std::move(e), j...);
      }
    }


    template<typename Operation, typename Arg, typename Count, typename...J>
    inline void unary_operation_in_place_impl(const Operation& operation, Arg& arg, const Count& count, J...j)
    {
      constexpr auto n = sizeof...(J);
      if constexpr (n < Count::value)
      {
        for (std::size_t i = 0; i < get_index_extent<n>(arg); ++i)
          unary_operation_in_place_impl(operation, arg, count, j..., i);
      }
      else
      {
        do_elem_operation_in_place(operation, arg, j...);
      }
    }

  }


  /**
   * \brief Perform a component-wise, in-place unary operation.
   * \details Examples:
   * - In-place unary operation without indices:
   *   \code
   *     auto m23 = make_dense_object_from<Eigen::Matrix<double, 3, 2>>(1, 2, 3, 4, 5, 6);
   *     auto opa = [](auto& arg){return ++arg;};
   *     std::cout << n_ary_operation(opa, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     2, 3,
   *     4, 5,
   *     6, 7
   *   \endcode
   * - In-place unary operation with indices:
   *   \code
   *     auto opb = [](auto& arg, std::size_t row, std::size_t col){arg += row + col;};
   *     std::cout << n_ary_operation(opb, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * \tparam Operation The n-ary operation taking an argument and, optionally, a set of indices.
   * The argument to the operation is a component of Arg and will be assignable as a non-const lvalue reference.
   * The operation may either return the result (as a value or reference) or return void
   * (in which case any changes to the component argument will be treated as an in-place modification).
   * \tparam Arg The argument, which must be non-const and writable.
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
  #ifdef __cpp_concepts
  template<typename Operation, writable Arg> requires detail::n_ary_operator<Operation, index_count_v<Arg>, Arg>
  #else
  template<typename Operation, typename Arg, std::enable_if_t<writable<Arg> and
    detail::n_ary_operator<Operation, index_count_v<Arg>, Arg>, int> = 0>
  #endif
  constexpr decltype(auto)
  unary_operation_in_place(const Operation& operation, Arg&& arg)
  {
    // \todo If the native library has its own facilities for doing this, use them.

    detail::unary_operation_in_place_impl(operation, arg, count_indices(arg));
    return std::forward<Arg>(arg);
  }


}

#endif
