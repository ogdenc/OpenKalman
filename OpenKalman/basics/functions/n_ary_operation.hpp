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

namespace OpenKalman
{
  // ----------------- //
  //  n_ary_operation  //
  // ----------------- //

  namespace detail
  {
    template<typename Op, typename...Args, std::size_t...I>
    constexpr bool is_invocable_with_indices(std::index_sequence<I...>)
    {
      return std::is_invocable_v<Op, Args&&..., decltype(I)...>;
    }


  template<typename Op, std::size_t...I, typename...Args>
  constexpr decltype(auto) n_ary_invoke_op(Op&& op, std::index_sequence<I...> seq, Args&&...args)
  {
    if constexpr (is_invocable_with_indices<Op&&, Args...>(seq))
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
    std::is_invocable_v<Op, Args&&...>
  struct n_ary_operator_traits_impl<Op, indices, Args...>
#else
  template<typename Op, std::size_t indices, typename...Args>
  struct n_ary_operator_traits_impl<Op, indices, std::enable_if_t<
    is_invocable_with_indices<Op, Args...>(std::make_index_sequence<indices> {}) or
    std::is_invocable_v<Op, Args&&...>>, Args...>
#endif
  {
    using type = decltype(n_ary_invoke_op(
      std::declval<Op>(), std::make_index_sequence<indices> {}, std::declval<Args&&>()...));
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
      (std::is_invocable<Op, typename std::add_lvalue_reference<typename scalar_type_of<Args>::type>::type...>::value or
        is_invocable_with_indices<Op, typename std::add_lvalue_reference<typename scalar_type_of<Args>::type>::type...>(
          std::make_index_sequence<Indices> {}))>, Args...>
    : std::true_type {};
#endif


    template<typename Op, std::size_t Indices, typename...Args>
#ifdef __cpp_concepts
    concept n_ary_operator = (std::is_invocable_v<Op, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...> or
        is_invocable_with_indices<Op, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(
          std::make_index_sequence<Indices> {}));
#else
    constexpr bool n_ary_operator = n_ary_operator_impl<Op, Indices, void, Args...>::value;
#endif


#ifdef __cpp_concepts
    template<typename T, typename DTup, typename Op, typename...Args>
#else
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
#endif
    struct interface_defines_n_ary_operation : std::false_type {};


    template<typename T, typename DTup, typename Op, typename...Args>
#ifdef __cpp_concepts
    requires requires(const DTup& d_tup, Op op, Args...args) {
      interface::LibraryRoutines<std::decay_t<T>>::n_ary_operation(d_tup, op, std::forward<Args>(args)...);
    }
    struct interface_defines_n_ary_operation<T, DTup, Op, Args...>
#else
    struct interface_defines_n_ary_operation<T, DTup, Op, std::void_t<
      decltype(interface::LibraryRoutines<std::decay_t<T>>::n_ary_operation_with_indices(
        std::declval<const DTup&>(), std::declval<const Op&>(), std::declval<Args>()...))>, Args...>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename T, typename DTup, typename Op, typename...Args>
#else
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
#endif
    struct interface_defines_n_ary_operation_with_indices : std::false_type {};


    template<typename T, typename DTup, typename Op, typename...Args>
#ifdef __cpp_concepts
    requires requires(const DTup& d_tup, const Op& op, Args...args) {
      interface::LibraryRoutines<std::decay_t<T>>::n_ary_operation_with_indices(d_tup, op, std::forward<Args>(args)...);
    }
    struct interface_defines_n_ary_operation_with_indices<T, DTup, Op, Args...>
#else
    struct interface_defines_n_ary_operation_with_indices<T, DTup, Op, std::void_t<
      decltype(interface::LibraryRoutines<std::decay_t<T>>::n_ary_operation_with_indices(
        std::declval<const DTup&>(), std::declval<const Op&>(), std::declval<Args>()...))>, Args...>
#endif
      : std::true_type {};


    template<typename Arg, std::size_t...I, typename...J>
    inline auto n_ary_operation_get_element_impl(Arg&& arg, std::index_sequence<I...>, J...j)
    {
      if constexpr (sizeof...(I) == sizeof...(J))
        return get_element(std::forward<Arg>(arg), (j < get_index_dimension_of<I>(arg) ? j : 0)...);
      else
        return get_element(std::forward<Arg>(arg), [](auto dim, const auto& j_tup){
          auto j = std::get<I>(j_tup);
          if (j < dim) return j;
          else return 0;
        }(get_index_dimension_of<I>(arg), std::tuple {j...})...);
    }


    template<typename Op, typename ArgsTup, std::size_t...ArgI, typename...J>
    inline auto n_ary_operation_get_element(const Op& op, ArgsTup&& args_tup, std::index_sequence<ArgI...>, J...j)
    {
      if constexpr (std::is_invocable_v<const Op&, scalar_type_of_t<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>..., J...>)
        return op(n_ary_operation_get_element_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<max_indices_of_v<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>> {},
          j...)..., j...);
      else
        return op(n_ary_operation_get_element_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<max_indices_of_v<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>> {},
          j...)...);
    }


    template<typename M, typename Op, typename ArgsTup, typename...J>
    inline void n_ary_operation_iterate(M& m, const Op& op, ArgsTup&& args_tup, std::index_sequence<>, J...j)
    {
      std::make_index_sequence<std::tuple_size_v<ArgsTup>> seq;
      set_element(m, n_ary_operation_get_element(op, std::forward<ArgsTup>(args_tup), seq, j...), j...);
    }


    template<typename M, typename Op, typename ArgsTup, std::size_t I, std::size_t...Is, typename...J>
    inline void n_ary_operation_iterate(M& m, const Op& op, ArgsTup&& args_tup, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(m); i++)
        n_ary_operation_iterate(m, op, std::forward<ArgsTup>(args_tup), std::index_sequence<Is...> {}, j..., i);
    }


    template<typename PatternMatrix, typename...Ds, typename Op, typename...Args>
    static constexpr auto
    n_ary_operation_impl(const std::tuple<Ds...>& d_tup, const Op& op, Args&&...args)
    {
      // constant_matrix:
      constexpr std::index_sequence_for<Ds...> seq {};
      if constexpr ((constant_matrix<Args> and ...) and not is_invocable_with_indices<Op&&, scalar_type_of_t<Args>...>(seq))
      {
        internal::scalar_constant_operation c {op, constant_coefficient {std::forward<Args>(args)}...};
        return std::apply(
          [](auto&&...as){ return make_constant_matrix_like<PatternMatrix>(std::forward<decltype(as)>(as)...); },
          std::tuple_cat(std::tuple{std::move(c)}, d_tup));
      }
      // other cases:
      /*//Note: these might not be necessary. n_ary_operation_iterate might be more efficient than library-defined operations.
      else if constexpr (is_invocable_with_indices<Op&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(seq) and
        detail::interface_defines_n_ary_operation_with_indices<PatternMatrix, std::tuple<Ds...>, Op&&, Args&&...>::value)
      {
        using Trait = interface::LibraryRoutines<std::decay_t<PatternMatrix>>;
        return Trait::n_ary_operation_with_indices(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }
      else if constexpr (is_invocable_with_indices<Op&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(seq) and
        detail::interface_defines_n_ary_operation<PatternMatrix, std::tuple<Ds...>, Op&&, Args&&...>::value)
      {
        using Trait = interface::LibraryRoutines<std::decay_t<PatternMatrix>>;
        return Trait::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }*/
      else
      {
        using Scalar = std::decay_t<typename n_ary_operator_traits<Op, sizeof...(Ds),
          std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>::type>;

        if constexpr (((dimension_size_of_v<Ds> == 1) and ...) and (element_gettable<Args&&, 2> and ...))
        {
          // one-by-one matrix
          auto e = op(get_element(std::forward<Args>(args), std::size_t(0), std::size_t(0))...);
          return make_dense_writable_matrix_from<PatternMatrix, Scalar>(d_tup, e);
        }
        else
        {
          auto m = std::apply([](auto&&...ds){
            return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(std::forward<decltype(ds)>(ds)...);
          }, d_tup);
          std::index_sequence_for<Ds...> seq;
          n_ary_operation_iterate(m, op, std::forward_as_tuple(std::forward<Args>(args)...), seq);
          return m;
        }
      }
    }


    // Check that dimensions of arguments Args are compatible with index descriptors Ds.
    template<std::size_t...ixs, typename DTup, typename...Args>
    constexpr void check_n_ary_dims(std::index_sequence<ixs...>, const DTup& d_tup, const Args&...args)
    {
      return ([](const DTup& d_tup, const Args&...args){
        constexpr auto ix = ixs;
        return ([](const DTup& d_tup, const auto& arg){
          using Arg = decltype(arg);
          if constexpr (dynamic_dimension<Arg, ix> or dynamic_index_descriptor<std::tuple_element_t<ix, DTup>>)
          {
            auto arg_d = get_index_descriptor<ix>(arg);
            auto tup_d = std::get<ix>(d_tup);
            auto dim_arg_d = get_dimension_size_of(arg_d);
            auto dim_tup_d = get_dimension_size_of(tup_d);
            if (not (arg_d == tup_d) and not internal::is_uniform_component_of(arg_d, tup_d))
              throw std::logic_error {"In an argument to n_ary_operation, the dimension of index " +
                std::to_string(ix) + " is " + std::to_string(dim_arg_d) + ", but should be 1 " +
                (dim_tup_d == 1 ? "" : "or " + std::to_string(dim_tup_d)) +
                "(the dimension of Ds number " + std::to_string(ix)};
          }
          else
          {
            using D_Arg = index_descriptor_of_t<Arg, ix>;
            using D = std::tuple_element_t<ix, DTup>;
            static_assert(equivalent_to<D_Arg, D> or equivalent_to_uniform_dimension_type_of<D_Arg, D> or
              (ix >= max_indices_of_v<Arg> and has_uniform_dimension_type<D>),
              "In argument to n_ary_operation, the dimension of each index must be either 1 or that of Ds.");
          }
        }(d_tup, args),...);
      }(d_tup, args...),...);
    }

  } // namespace detail


  /**
   * \brief Perform a component-wise n-ary operation, using broadcasting to match the size of a pattern matrix.
   * \details This overload is for unary, binary, and higher n-ary operations. Examples:
   * - Unary operation, no broadcasting:
   *   \code
   *     auto ds32 = std::tuple {Dimensions<3>{}, Dimensions<2>{}};
   *     auto op1 = [](auto arg){return 3 * arg;};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto m32 = make_dense_writable_matrix_from<M>(ds32, 1, 2, 3, 4, 5, 6);
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
   *     auto m31 = make_dense_writable_matrix_from<M>(ds31, 1, 2, 3);
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
   * \tparam Ds Index descriptors defining the size of the result.
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices indicating the location
   * within the result. The operation must return a scalar value.
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<index_descriptor...Ds, typename Operation, indexible...Args> requires (sizeof...(Args) > 0) and
    (sizeof...(Ds) >= std::max({max_indices_of_v<Args>...})) and detail::n_ary_operator<Operation, sizeof...(Ds), Args...>
#else
  template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<
    (index_descriptor<Ds> and ...) and (indexible<Args> and ...) and
    (sizeof...(Args) > 0) and (sizeof...(Ds) >= std::max({max_indices_of<Args>::value...})) and
    detail::n_ary_operator<Operation, sizeof...(Ds), Args...>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(const std::tuple<Ds...>& d_tup, const Operation& operation, Args&&...args)
  {
    detail::check_n_ary_dims(std::index_sequence_for<Ds...> {}, d_tup, args...);
    using Arg0 = std::tuple_element_t<0, std::tuple<Args...>>; // \todo Pick the first appropriate pattern matrix, even if not the first one.
    return detail::n_ary_operation_impl<Arg0>(d_tup, operation, std::forward<Args>(args)...);
  }


  namespace detail
  {
    template<std::size_t ix, typename Arg, typename...Args>
    constexpr auto find_max_dim(const Arg& arg, const Args&...args)
    {
      if constexpr (sizeof...(Args) == 0) return get_index_descriptor<ix>(arg);
      else
      {
        auto max_d = find_max_dim<ix>(args...);
        using Arg_D = index_descriptor_of_t<Arg, ix>;
        using Max_D = decltype(max_d);
        if constexpr (fixed_index_descriptor<Arg_D> and fixed_index_descriptor<Max_D>)
        {
          constexpr auto dim_arg_d = dimension_size_of_v<Arg_D>;
          if constexpr (equivalent_to<Arg_D, Max_D> or
            (dim_arg_d == 1 and equivalent_to_uniform_dimension_type_of<Arg_D, Max_D>))
          {
            return max_d;
          }
          else
          {
            constexpr auto dim_max_d = dimension_size_of_v<Max_D>;
            static_assert(dim_max_d != 1 or not equivalent_to_uniform_dimension_type_of<Max_D, Arg_D>,
              "The dimension of arguments to n_ary_operation are not compatible with each other for at least one index.");
            return get_index_descriptor<ix>(arg);
          }
        }
        else if constexpr (euclidean_index_descriptor<Arg_D> and euclidean_index_descriptor<Max_D>)
        {
          auto arg_d = get_index_descriptor<ix>(arg);
          auto dim_arg_d = get_dimension_size_of(arg_d);
          auto dim_max_d = get_dimension_size_of(max_d);
          if (dim_arg_d == dim_max_d or (dim_arg_d == 1 and dim_arg_d <= dim_max_d)) return dim_max_d;
          else if (dim_max_d == 1 and dim_max_d <= dim_arg_d) return dim_arg_d;
          else throw std::invalid_argument {"The dimension of arguments to n_ary_operation are not compatible with "
            "each other for at least index " + std::to_string(ix) + "."};
        }
        else
        {
          auto arg_d = get_index_descriptor<ix>(arg);
          using Scalar = scalar_type_of_t<Arg>;
          if (internal::is_uniform_component_of(arg_d, max_d))
          {
            if constexpr (internal::is_DynamicTypedIndex<Max_D>::value) return DynamicTypedIndex {max_d};
            else return DynamicTypedIndex<Scalar> {max_d};
          }
          else if (internal::is_uniform_component_of(max_d, arg_d))
          {
            if constexpr (internal::is_DynamicTypedIndex<Max_D>::value) return DynamicTypedIndex {arg_d};
            else return DynamicTypedIndex<Scalar> {arg_d};
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

  } // namespace detail


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
   *     auto m31 = make_dense_writable_matrix_from<M>(ds31, 1, 2, 3);
   *     auto m32 = make_dense_writable_matrix_from<M>(ds32, 1, 2, 3, 4, 5, 6);
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
    detail::n_ary_operator<Operation, std::max({max_indices_of_v<Args>...}), Args...>
#else
  template<typename Operation, typename...Args, std::enable_if_t<(indexible<Args> and ...) and
    (sizeof...(Args) > 0) and detail::n_ary_operator<Operation, std::max({max_indices_of_v<Args>...}), Args...>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(const Operation& operation, Args&&...args)
  {
    auto d_tup = detail::find_max_dims(std::make_index_sequence<std::max({max_indices_of_v<Args>...})> {}, args...);
    using Arg0 = std::tuple_element_t<0, std::tuple<Args...>>;
    return detail::n_ary_operation_impl<Arg0>(std::move(d_tup), operation, std::forward<Args>(args)...);
  }


  namespace detail
  {
    template<typename PatternMatrix, typename Operation, typename Descriptors_tuple, typename Index_seq, typename K_seq, typename...Is>
    void nullary_set_elements(PatternMatrix& m, Operation&& op, const Descriptors_tuple&, Index_seq, K_seq, Is...is)
    {
      constexpr auto seq = std::make_index_sequence<max_indices_of_v<PatternMatrix>> {};
      if constexpr (detail::is_invocable_with_indices<Operation&&>(seq))
        set_element(m, op(is...), is...);
      else
        set_element(m, op(), is...);
    }


    template<std::size_t DsIndex, std::size_t...DsIndices, typename PatternMatrix, typename Operation,
      typename Descriptors_tuple, std::size_t...indices, std::size_t...Ks, typename...Is>
    void nullary_set_elements(PatternMatrix& m, Operation&& op, const Descriptors_tuple& ds_tup,
      std::index_sequence<indices...> index_seq, std::index_sequence<Ks...> k_seq, Is...is)
    {
      if constexpr (((DsIndex == indices) or ...))
      {
        constexpr std::integral_constant<size_t, ((DsIndex == indices ? Ks : 0) + ...)> i;
        nullary_set_elements<DsIndices...>(m, std::forward<Operation>(op), ds_tup, index_seq, k_seq, is..., i);
      }
      else
      {
        // Iterate through the dimensions of the current DsIndex and add set elements for each dimension iteratively.
        for (std::size_t i = 0; i < get_dimension_size_of(std::get<DsIndex>(ds_tup)); ++i)
        {
          nullary_set_elements<DsIndices...>(m, op, ds_tup, index_seq, k_seq, is..., i);
        }
      }
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, typename PatternMatrix, typename Operations_tuple,
      typename Descriptors_tuple, typename UniqueIndicesSeq, std::size_t...AllDsIndices, typename K_seq>
    void nullary_iterate(PatternMatrix& m, Operations_tuple&& op_tup, const Descriptors_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, std::index_sequence<AllDsIndices...>, K_seq k_seq)
    {
      nullary_set_elements<AllDsIndices...>(m, std::get<CurrentOpIndex>(
        std::forward<Operations_tuple>(op_tup)), ds_tup, unique_indices_seq, k_seq);
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, std::size_t index, std::size_t...indices,
      typename PatternMatrix, typename Operations_tuple, typename Descriptors_tuple, typename UniqueIndicesSeq, typename AllDsSeq,
      std::size_t...Ks, std::size_t...Js, typename...J_seqs>
    void nullary_iterate(PatternMatrix& m, Operations_tuple&& op_tup, const Descriptors_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, AllDsSeq all_ds_seq, std::index_sequence<Ks...>, std::index_sequence<Js...>,
      J_seqs...j_seqs)
    {
      constexpr std::size_t new_factor = factor / dimension_size_of_v<std::tuple_element_t<index, Descriptors_tuple>>;

      ((nullary_iterate<CurrentOpIndex + new_factor * Js, new_factor, indices...>(
        m, std::forward<Operations_tuple>(op_tup), ds_tup, unique_indices_seq, all_ds_seq,
        std::index_sequence<Ks..., Js>{}, j_seqs...)),...);
    }

  } // namespace detail


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
   * \tparam Ds Index descriptors for each index of the result. Index descriptors corresponding to indices must be
   * of a \ref fixed_index_descriptor type.
   * \tparam Operations The nullary operations. The number of operations must equal the product of each dimension Ds
   * corresponding to indices. The order of the operations depends on the order of indices, with the left-most index
   * being the most major, and the right-most index being the most minor. For example, if indices
   * are {0, 1} for a matrix result, the operations must be in row-major order. If the indices are {1, 0}, the
   * operations must be in column-major order.
   * Each operation may be invocable with no arguments or invocable with as many indices
   * as there are index descriptors Ds. (If the index corresponds to one of designaged <code>indices</code>, then the
   * operation will be called with <code>std::integral_constant<std::size_t, _index_>{}</code> for that index,
   * instead of <code>std::size_t</code>.))
   * \return A matrix or array in which each component is the result of calling Operation and which has
   * dimensions corresponding to Ds
   */
  #ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, index_descriptor...Ds, typename...Operations>
  requires ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == (1 * ... * dimension_size_of_v<std::tuple_element_t<indices, std::tuple<Ds...>>>)) and
    (detail::n_ary_operator<Operations, max_indices_of_v<PatternMatrix>> and ...)
  #else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == (1 * ... * dimension_size_of<std::tuple_element_t<indices, std::tuple<Ds...>>>::value)) and
    (detail::n_ary_operator<Operations, max_indices_of_v<PatternMatrix>> and ...), int> = 0>
  #endif
  constexpr auto
  n_ary_operation(const std::tuple<Ds...>& d_tup, const Operations&...operations)
  {
    using Scalar = std::common_type_t<std::decay_t<
      typename detail::n_ary_operator_traits<Operations, max_indices_of_v<PatternMatrix>>::type>...>;

    // One operation for all elements combined:
    if constexpr (sizeof...(Operations) == 1)
    {
      return detail::n_ary_operation_impl<PatternMatrix>(d_tup, operations...);
    }
    // One operation for each element, and the operations are not invocable with indices:
    else if constexpr (((not dynamic_index_descriptor<Ds>) and ...) and
      sizeof...(operations) == (dimension_size_of_v<Ds> * ...) and
      not (detail::is_invocable_with_indices<const Operations&>(std::make_index_sequence<max_indices_of_v<PatternMatrix>> {}) or ...))
    {
      return make_dense_writable_matrix_from<PatternMatrix, Scalar>(d_tup, operations()...);
    }
    // All other cases:
    else
    {
      auto m = std::apply([](const auto&...ds){ return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(ds...); }, d_tup);
      auto operations_tuple = std::forward_as_tuple(operations...);
      detail::nullary_iterate<0, sizeof...(Operations), indices...>(
        m, std::move(operations_tuple), d_tup,
        std::index_sequence<indices...> {},
        std::index_sequence_for<Ds...> {},
        std::index_sequence<> {},
        std::make_index_sequence<dimension_size_of_v<std::tuple_element_t<indices, std::tuple<Ds...>>>> {}...);
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
   * Dimensions of PatternMatrix corresponding to indices must be of a \ref fixed_index_descriptor type.
   * \tparam indices The indices, if any, along which there will be a different operator for each element along that index.
   * \tparam Operations The nullary operations. The number of operations must equal the product of each dimension Ds
   * corresponding to indices.
   * \return A matrix or array in which each component is the result of calling Operation and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, index_descriptor...Ds, typename...Operations>
  requires ((fixed_index_descriptor<index_descriptor_of_t<PatternMatrix, indices>>) and ...) and
    (sizeof...(Operations) == (1 * ... * index_dimension_of_v<PatternMatrix, indices>))
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    ((fixed_index_descriptor<typename index_descriptor_of<PatternMatrix, indices>::type>) and ...) and
    (sizeof...(Operations) == (1 * ... * index_dimension_of<PatternMatrix, indices>::value)), int> = 0>
#endif
  constexpr auto
  n_ary_operation(Operations&&...operations)
  {
    auto d_tup = get_all_dimensions_of<PatternMatrix>();
    return n_ary_operation<PatternMatrix, indices...>(d_tup, std::forward<Operations>(operations)...);
  }

  // -------------------------- //
  //  n_ary_operation_in_place  //
  // -------------------------- //

  namespace detail
  {
    template<typename Arg, std::size_t...I>
    constexpr decltype(auto) n_ary_get_element_0(Arg&& arg, std::index_sequence<I...>)
    {
      return get_element(arg, static_cast<decltype(I)>(0)...);
    }


    template<typename Operation, typename Arg, typename...J>
    inline void unary_operation_in_place_impl(const Operation& operation, Arg&& arg, std::index_sequence<>, J...j)
    {
      if constexpr (std::is_invocable_v<const Operation&, std::add_lvalue_reference_t<scalar_type_of_t<Arg>>, J...>)
        operation(get_element(arg, j...), j...);
      else
        operation(get_element(arg, j...));
    }

    template<typename Operation, typename Arg, std::size_t I, std::size_t...Is, typename...J>
    inline void unary_operation_in_place_impl(const Operation& operation, Arg&& arg, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(arg); ++i)
        unary_operation_in_place_impl(operation, arg, std::index_sequence<Is...> {}, j..., i);
    }
  } // namespace detail


  /**
   * \brief Perform a component-wise, in-place unary operation.
   * \details Examples:
   * - In-place unary operation without indices:
   *   \code
   *     auto m23 = make_dense_writable_matrix_from<Eigen::Matrix<double, 3, 2>>(1, 2, 3, 4, 5, 6);
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
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices.
   * The result of this operation is ignored, and all that matters are the side-effects.
   * \tparam Arg The argument, which must be non-const and writable.
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
  #ifdef __cpp_concepts
  template<typename Operation, writable Arg> requires detail::n_ary_operator<Operation, max_indices_of_v<Arg>, Arg>
  #else
  template<typename Operation, typename Arg, std::enable_if_t<writable<Arg> and
    detail::n_ary_operator<Operation, max_indices_of_v<Arg>, Arg>, int> = 0>
  #endif
  constexpr decltype(auto)
  unary_operation_in_place(const Operation& operation, Arg&& arg)
  {
    // \todo If the native library has its own facilities for doing this, use it.

    constexpr std::make_index_sequence<max_indices_of_v<Arg>> seq;
    using G = decltype(detail::n_ary_get_element_0(std::declval<Arg&&>(), seq));
    static_assert(std::is_same_v<G, std::decay_t<G>&>,
      "unary_operation_in_place requires get_element(arg) to return a non-const lvalue reference.");

    detail::unary_operation_in_place_impl(operation, arg, seq);
    return std::forward<Arg>(arg);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_N_ARY_OPERATION_HPP
