/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for ConstantAdapter
 */

#ifndef OPENKALMAN_CONSTANTADAPTER_HPP
#define OPENKALMAN_CONSTANTADAPTER_HPP

#include <type_traits>
#include <algorithm>

namespace OpenKalman
{
  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

#ifdef __cpp_concepts
  template<indexible PatternMatrix, value::scalar Scalar, auto...constant>
    requires (sizeof...(constant) == 0) or requires { Scalar {constant...}; }
#else
  template<typename PatternMatrix, typename Scalar, auto...constant>
#endif
  struct ConstantAdapter : internal::library_base_t<ConstantAdapter<PatternMatrix, Scalar, constant...>, PatternMatrix>
  {

  private:

#ifndef __cpp_concepts
    static_assert(indexible<PatternMatrix>);
    static_assert(value::scalar<Scalar>);
    static_assert(sizeof...(constant) == 0 or std::is_constructible_v<Scalar, decltype(constant)...>);
#endif

    using MyConstant = std::conditional_t<sizeof...(constant) == 0, Scalar, value::Fixed<Scalar, constant...>>;
    using MyScalarType = value::number_type_of_t<MyConstant>;

    using AllDescriptorsType = decltype(all_vector_space_descriptors(std::declval<PatternMatrix>()));
    using DescriptorCollection = std::conditional_t<
      pattern_tuple<AllDescriptorsType>,
      AllDescriptorsType, 
      std::vector<DynamicDescriptor<MyScalarType>>>;


    template<typename D, std::size_t...Ix, std::size_t...IxExtra>
    static constexpr auto make_descriptors_tuple(D&& d, std::index_sequence<Ix...>, std::index_sequence<IxExtra...>)
    {
      if constexpr ((... or dynamic_pattern<std::tuple_element_t<IxExtra, D>>))
        if ((... or (std::get<IxExtra>(d) != Axis{})));
          throw std::invalid_argument {"Too many elements in vector space descriptors_collection_tuple argument of a constant adapter"};

      return std::tuple {[](D&& d){
        using E = std::tuple_element_t<Ix, DescriptorCollection>;

        if constexpr (Ix < std::tuple_size_v<D>)
        {
          if constexpr (fixed_pattern<E>)
          {
            if constexpr (dynamic_pattern<std::tuple_element_t<Ix, D>>)
              if (std::get<Ix>(d) != E{}) throw std::invalid_argument {
                "Invalid dynamic element in pattern_tuple argument of a constant adapter"};
            return E{};
          }
          else // if constexpr (dynamic_pattern<E>)
          {
            return std::forward<D>(d);
          }
        }
        else
        {
          return E{};
        }
      }(std::forward<D>(d))...};
    }


    template<std::size_t N = 0, typename Iterator, typename Sentinel>
    static constexpr auto make_descriptors_tuple_from_range(Iterator it, const Sentinel& end)
    {
      if constexpr (N >= std::tuple_size_v<DescriptorCollection>)
      {
        for (; it != end; ++it) if (*it != Axis{}) throw std::invalid_argument {
          "Too many elements in vector space descriptors_collection range argument of a constant adapter"};
        return std::tuple {};
      }
      else if constexpr (fixed_pattern<std::tuple_element_t<N, DescriptorCollection>>)
      {
        using E = std::tuple_element_t<N, DescriptorCollection>;
        if (it == end)
        {
          if (not compares_with<E, Axis>) throw std::invalid_argument {
            "Too few elements in vector space descriptors_collection range argument of a constant adapter"};
          return std::tuple_cat(std::forward_as_tuple(E{}),
            make_descriptors_tuple_from_range<N + 1>(std::move(it), end));
        }
        else
        {
          if constexpr (dynamic_pattern<decltype(*it)>)
            if (*it != E{}) throw std::invalid_argument {
              "Invalid dynamic element in pattern_collection range argument of a constant adapter"};
          return std::tuple_cat(std::forward_as_tuple(E{}),
            make_descriptors_tuple_from_range<N + 1>(++it, end));
        }
      }
      else // if constexpr (dynamic_pattern<std::tuple_element_t<N, DescriptorCollection>>)
      {
        using E = std::tuple_element_t<N, DescriptorCollection>;
        if (it == end) throw std::invalid_argument {
          "Too few elements in vector space descriptors_collection range argument of a constant adapter"};
        std::tuple_element_t<N, DescriptorCollection> e {*it};
        return std::tuple_cat(std::forward_as_tuple(std::move(e)),
          make_descriptors_tuple_from_range<N + 1>(++it, end));
      }
    }


    template<typename D, std::size_t...Ix>
    static constexpr auto make_descriptors_range_from_tuple(D&& d, std::index_sequence<Ix...>)
    {
      return DescriptorCollection {{DynamicDescriptor<MyScalarType> {std::get<Ix>(std::forward<D>(d))}...}};
    }
    
    
    template<typename Descriptors>
    static constexpr auto make_descriptors_collection(Descriptors&& descriptors)
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      if constexpr (pattern_tuple<DescriptorCollection>)
      {
        if constexpr (pattern_tuple<Descriptors>)
        {
          constexpr auto N = std::tuple_size_v<DescriptorCollection>;
          constexpr auto M = std::tuple_size_v<Descriptors>;
          return make_descriptors_tuple(std::forward<Descriptors>(descriptors),
            std::make_index_sequence<N> {},
            std::make_index_sequence<(M > N) ? M - N : 0> {});
        }
        else
        {
          return make_descriptors_tuple_from_range(ranges::begin(descriptors), ranges::end(descriptors));
        }
      }
      else if constexpr (pattern_tuple<Descriptors>)
      {
        return make_descriptors_range_from_tuple(std::forward<Descriptors>(descriptors),
          std::make_index_sequence<std::tuple_size_v<Descriptors>> {});
      }
      else
      {
        DescriptorCollection ret;
        std::copy(ranges::begin(descriptors), ranges::end(descriptors), ranges::begin(ret));
        return ret;
      }
    }

  public:

    /**
     * \brief Construct from \ref value::scalar and a \ref pattern_collection
     */
#ifdef __cpp_lib_ranges
    template<value::scalar C, pattern_collection Descriptors> requires
      std::constructible_from<MyConstant, C&&> and 
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>
#else
    template<typename C, typename Descriptors, std::enable_if_t<
      value::scalar<C> and pattern_collection<Descriptors> and
      std::is_constructible_v<MyConstant, C&&> and 
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Descriptors&& descriptors) 
      : my_constant {std::forward<C>(c)}, 
        descriptor_collection {make_descriptors_collection(std::forward<Descriptors>(descriptors))} {}


    /**
     * \overload
     * \brief Same as above, where the constant is known at compile time.
     */
#ifdef __cpp_lib_ranges
    template<pattern_collection Descriptors> requires
      value::fixed<MyConstant> and
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>
#else
    template<typename Descriptors, std::enable_if_t<
      pattern_collection<Descriptors> and
      value::fixed<MyConstant> and
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Descriptors&& descriptors) 
      : descriptor_collection {make_descriptors_collection(std::forward<Descriptors>(descriptors))} {}

  
    /**
     * \brief Construct based on a \ref value::scalar and the shape of an \ref indexible reference object.
     * \details 
     * The following constructs a 2-by-3 ConstantAdapter with constant 5.
     * \code
     * ConstantAdapter {std::integral_constant<int, 5>{}, eigen_matrix_t<double, 2, 3>{})
     * ConstantAdapter {std::integral_constant<int, 5>{}, eigen_matrix_t<double, 2, Eigen::Dynamic>(2, 3))
     * ConstantAdapter {5, eigen_matrix_t<double, 2, 3>{})
     * ConstantAdapter {5, eigen_matrix_t<double, Eigen::Dynamic, 3>(2, 3))
     * \endcode
     */
#ifdef __cpp_concepts
    template<value::scalar C, vector_space_descriptors_may_match_with<PatternMatrix> Arg> requires
      std::constructible_from<MyConstant, C&&>
#else
    template<typename C, typename Arg, std::enable_if_t<
      value::scalar<C> and vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::is_constructible_v<MyConstant, C&&>, int> = 0>
#endif
    constexpr ConstantAdapter(C&& c, const Arg& arg) :
      my_constant {std::forward<C>(c)}, 
      descriptor_collection {make_descriptors_collection(all_vector_space_descriptors(arg))} {}


    /**
     * \overload 
     * \brief Same as above, where the constant is known at compile time.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<PatternMatrix> Arg> requires
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      value::fixed<MyConstant>
#else
    template<typename Arg, std::enable_if_t<
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and 
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      value::fixed<MyConstant>, int> = 0>
#endif
    constexpr ConstantAdapter(const Arg& arg) :
      descriptor_collection {make_descriptors_collection(all_vector_space_descriptors(arg))} {}


    /**
     * \brief Construct from another constant object.
     */
#ifdef __cpp_concepts
    template<constant_matrix Arg> requires
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::constructible_from<MyConstant, constant_coefficient<Arg>>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg> and
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::is_constructible_v<MyConstant, constant_coefficient<Arg>>, int> = 0>
#endif
    constexpr ConstantAdapter(const Arg& arg) :
      my_constant {constant_coefficient {arg}}, 
      descriptor_collection {make_descriptors_collection(all_vector_space_descriptors(arg))} {}


    /**
     * \brief Construct using a full set of \ref coordinate::pattern objects.
     * \details For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>(std::integral_constant<int, 5>{}, 2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, 3>>(5., 2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>>(5., 2, 3)
     * \endcode
     */
#ifdef __cpp_concepts
    template<value::scalar C, coordinate::pattern...Ds> requires
      std::constructible_from<MyConstant, C&&> and
      compatible_with_vector_space_descriptor_collection<PatternMatrix, std::tuple<Ds...>>
#else
    template<typename C, typename...Ds, std::enable_if_t<
      value::scalar<C> and (coordinate::pattern<Ds> and ...) and
      std::is_constructible_v<MyConstant, C&&> and
      compatible_with_vector_space_descriptor_collection<PatternMatrix, std::tuple<Ds...>>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Ds&&...ds)
      : ConstantAdapter(std::forward<C>(c), std::tuple {std::forward<Ds>(ds)...}) {}


    /**
     * \overload
     * \brief Same as above, where the constant is known at compile time.
     * \tparam Ds A set of \ref coordinate::pattern objects corresponding to PatternMatrix.
     * \details
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 1, 1>, 5>()
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>(2, std::integral_constant<int, 3>{})
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, 3>, 5>(2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>, 5>(2, 3)
     * \endcode
     */
#ifdef __cpp_concepts
    template<coordinate::pattern...Ds> requires
      value::fixed<MyConstant> and
      compatible_with_vector_space_descriptor_collection<PatternMatrix, std::tuple<Ds...>>
#else
    template<typename...Ds, std::enable_if_t<
      (coordinate::pattern<Ds> and ...) and
      value::fixed<MyConstant> and
      compatible_with_vector_space_descriptor_collection<PatternMatrix, std::tuple<Ds...>>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Ds&&...ds)
      : ConstantAdapter(std::tuple {std::forward<Ds>(ds)...}) {}

  private:

    template<std::size_t N = 0>
    constexpr auto make_dynamic_dimensions_tuple()
    {
      if constexpr (N < index_count_v<PatternMatrix>)
        return std::tuple_cat(std::forward_as_tuple(std::get<N>(descriptor_collection)), make_dynamic_dimensions_tuple<N + 1>());
      else
        return std::tuple {};
    }


    template<std::size_t N = 0, typename D, typename...Ds>
    constexpr auto make_dynamic_dimensions_tuple(D&& d, Ds&&...ds)
    {
      if constexpr (dynamic_pattern<std::tuple_element_t<N, DescriptorCollection>>)
        return std::tuple_cat(std::forward_as_tuple(std::forward<D>(d)),
          make_dynamic_dimensions_tuple<N + 1>(std::forward<Ds>(ds)...));
      else
        return std::tuple_cat(std::forward_as_tuple(std::get<N>(descriptor_collection)),
          make_dynamic_dimensions_tuple<N + 1>(std::forward<D>(d), std::forward<Ds>(ds)...));
    }

  public:

    /**
     * \brief Construct using only applicable \ref dynamic_pattern.
     * \tparam Ds A set of \ref dynamic_pattern corresponding to each of
     * class template parameter Ds that is dynamic, in order of Ds. This list should omit any \ref fixed_pattern.
     * \details If PatternMatrix has no dynamic dimensions, this is a default constructor.
     * The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(std::integral_constant<int, 5>{}, 3) // Fixed rows and dynamic columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>>(5) // Fixed rows and columns.
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, 3>>(5, 2) // Dynamic rows and fixed columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(5, 3) // Fixed rows and dynamic columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<value::scalar C, dynamic_pattern...Ds> requires
      pattern_tuple<DescriptorCollection> and
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      std::constructible_from<MyConstant, C&&>
#else
    template<typename C, typename...Ds, std::enable_if_t<
      value::scalar<C> and (dynamic_pattern<Ds> and ...) and
      pattern_tuple<DescriptorCollection> and
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      std::is_constructible_v<MyConstant, C&&>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Ds&&...ds) 
      : ConstantAdapter(std::forward<C>(c), make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)) {}


    /**
     * \overload
     * \brief Same as above, where the constant is known at compile time.
     * \details For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>(2, 3) // Dynamic rows and columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(3) // Fixed rows and dynamic columns.
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, 3>, 5>(2) // Dynamic rows and fixed columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>() // Fixed rows and columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<dynamic_pattern...Ds> requires
      pattern_tuple<DescriptorCollection> and
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      value::fixed<MyConstant>
#else
    template<typename...Ds, std::enable_if_t<
      value::scalar<C> and (dynamic_pattern<Ds> and ...) and
      pattern_tuple<DescriptorCollection> and
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      value::fixed<MyConstant>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Ds&&...ds) 
      : ConstantAdapter(make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)) {}


    /**
     * \brief Assign from a compatible \ref constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix Arg> requires
      (not std::derived_from<Arg, ConstantAdapter>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::assignable_from<MyConstant, constant_coefficient<Arg>>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg> and
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::is_assignable_v<MyConstant, constant_coefficient<Arg>>, int> = 0>
#endif
    constexpr auto& operator=(const Arg& arg)
    {
      if constexpr (not vector_space_descriptors_match_with<Arg, PatternMatrix>) 
        if (not vector_space_descriptors_match(*this, arg)) throw std::invalid_argument {
          "Argument to ConstantAdapter assignment operator has non-matching vector space descriptors."};
      my_constant = constant_coefficient {arg};
      return *this;
    }


    /**
     * \brief Comparison operator
     */
#ifdef __cpp_concepts
    template<indexible Arg>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
    constexpr bool operator==(const Arg& arg) const
    {
      if constexpr (not vector_space_descriptors_may_match_with<Arg, PatternMatrix>)
        return false;
      else if constexpr (constant_matrix<Arg>)
        return value::to_number(constant_coefficient{arg}) == value::to_number(my_constant) and vector_space_descriptors_match(*this, arg);
      else
      {
        auto c = to_native_matrix<PatternMatrix>(*this);
        static_assert(not std::is_same_v<decltype(c), ConstantAdapter>,
          "interface::library_interface<PatternMatrix>::to_native_matrix(*this) must define an object within the library of Arg");
        return std::move(c) == arg;
      }
    }


#ifndef __cpp_impl_three_way_comparison
    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<indexible Arg> (not constant_adapter<Arg>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not constant_adapter<Arg>, int> = 0>
#endif
    friend constexpr bool operator==(const Arg& arg, const ConstantAdapter& c)
    {
      if constexpr (not vector_space_descriptors_may_match_with<Arg, PatternMatrix>)
        return false;
      else if constexpr (constant_matrix<Arg>)
        return value::to_number(constant_coefficient{arg}) == value::to_number(c.get_scalar_constant()) and vector_space_descriptors_match(arg, c);
      else
      {
        auto new_c = to_native_matrix<Arg>(c);
        static_assert(not std::is_same_v<decltype(new_c), ConstantAdapter>,
          "interface::library_interface<Arg>::to_native_matrix(c) must define an object within the library of Arg");
        return arg == std::move(new_c);
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
    constexpr bool operator!=(const Arg& arg) const { return not (*this == arg); }


#ifdef __cpp_concepts
    template<indexible Arg> (not std::same_as<Arg, ConstantAdapter>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not std::is_same_v<Arg, ConstantAdapter>, int> = 0>
#endif
    friend constexpr bool operator!=(const Arg& arg, const ConstantAdapter& c) { return not (arg == c); }
#endif


    /**
     * \brief Access a component at a set of indices.
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_lib_ranges
    template<index_range_for<PatternMatrix> Indices> requires (not empty_object<PatternMatrix>)
    constexpr value::scalar auto
#else
    template<typename Indices, std::enable_if_t<
      index_range_for<Indices, PatternMatrix> and (not empty_object<PatternMatrix>), int> = 0>
    constexpr auto
#endif
    operator[](const Indices& indices) const 
    {
      return value::to_number(my_constant);
    }


    /**
     * \brief Get the \ref value::scalar associated with this object.
     */
#ifdef __cpp_concepts
    constexpr value::scalar auto
#else
    constexpr auto
#endif
    get_scalar_constant() const
    {
      return my_constant;
    }


    friend auto operator-(const ConstantAdapter& arg)
    {
      if constexpr (zero<ConstantAdapter>) return arg;
      else
      {
        value::operation op {std::negate{}, constant_coefficient{arg}};
        return make_constant(arg, op);
      }
    }


#ifdef __cpp_concepts
    friend auto operator*(const ConstantAdapter& arg, value::scalar auto s)
#else
    template<typename S, std::enable_if_t<value::scalar<S>, int> = 0>
    friend auto operator*(const ConstantAdapter& arg, S s)
#endif
    {
      if constexpr (zero<decltype(arg)>) return arg;
      else
      {
        return make_constant(arg, constant_coefficient{arg} * s);
      }
    }


#ifdef __cpp_concepts
    friend auto operator*(value::scalar auto s, const ConstantAdapter& arg)
#else
    template<typename S, std::enable_if_t<value::scalar<S>, int> = 0>
    friend auto operator*(S s, const ConstantAdapter& arg)
#endif
    {
      if constexpr (zero<decltype(arg)>) return arg;
      else
      {
        return make_constant(arg, s * constant_coefficient{arg});
      }
    }


#ifdef __cpp_concepts
    friend auto operator/(const ConstantAdapter& arg, value::scalar auto s)
#else
    template<typename S, std::enable_if_t<value::scalar<S>, int> = 0>
    friend auto operator/(const ConstantAdapter& arg, S s)
#endif
    {
      return make_constant(arg, constant_coefficient{arg} / s);
    }

  protected:

    MyConstant my_constant;

    DescriptorCollection descriptor_collection;

    friend struct interface::indexible_object_traits<ConstantAdapter>;
    friend struct interface::library_interface<ConstantAdapter>;

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<value::scalar C, indexible Arg>
#else
  template<typename C, typename Arg, std::enable_if_t<value::scalar<C> and indexible<Arg>, int> = 0>
#endif
  ConstantAdapter(const C&, const Arg&) -> ConstantAdapter<Arg, C>;


#ifdef __cpp_concepts
  template<constant_matrix Arg> requires (not constant_adapter<Arg>)
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg> and (not constant_adapter<Arg>), int> = 0>
#endif
  ConstantAdapter(const Arg&) -> ConstantAdapter<Arg, constant_coefficient<Arg>>;


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct indexible_object_traits<ConstantAdapter<PatternMatrix, Scalar, constant...>>
    {
    private:

      using XprType = ConstantAdapter<PatternMatrix, Scalar, constant...>;

    public:

      using scalar_type = typename XprType::MyScalarType;
      using MyDims = typename XprType::MyDimensions_t;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg)
      {
        if constexpr (index_count_v<PatternMatrix> == dynamic_size)
          return std::forward<Arg>(arg).descriptor_collection.size();
        else
          return std::tuple_size<MyDims>{};
      }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, const N& n)
      {
        if constexpr (index_count_v<PatternMatrix> == dynamic_size)
        {
          return std::forward<Arg>(arg).descriptor_collection[static_cast<typename MyDims::size_type>(n)];
        }
        else if constexpr (value::fixed<N>)
        {
          if constexpr (N::value >= index_count_v<PatternMatrix>) return Dimensions<1>{};
          else return std::get<N::value>(std::forward<Arg>(arg).descriptor_collection);
        }
        else if (n >= std::tuple_size_v<MyDims>)
        {
          return 1_uz;
        }
        else
        {
          return std::apply(
            [](auto&&...ds){ return std::array<std::size_t, std::tuple_size_v<MyDims>> {std::forward<decltype(ds)>(ds)...}; },
            std::forward<Arg>(arg).descriptor_collection)[n];
        }
      }


      // No nested_object defined


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg) { return arg.get_scalar_constant(); }


      // No get_constant_diagonal defined


      template<Applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<PatternMatrix, b>;


      template<Applicability b>
      static constexpr bool is_square = OpenKalman::square_shaped<PatternMatrix, b>;


      // No is_triangular, is_triangular_adapter, is_hermitian, or hermitian_adapter_type defined


      static constexpr bool is_writable = false;


      // No raw_data, layout, or strides defined.

    };


    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct library_interface<ConstantAdapter<PatternMatrix, Scalar, constant...>>
    {
      template<typename Derived>
      using LibraryBase = internal::library_base_t<Derived, PatternMatrix>;


      template<typename Arg, typename Indices>
      static constexpr auto
      get_component(Arg&& arg, const Indices&) { return std::forward<Arg>(arg).get_scalar_constant(); }


      // No set_component defined  because ConstantAdapter is not writable.


      template<typename Arg>
      static decltype(auto)
      to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<PatternMatrix>(std::forward<Arg>(arg));
      }


      template<Layout layout, typename S, typename D>
      static auto
      make_default(D&& d)
      {
        return make_dense_object<PatternMatrix, layout, S>(std::forward<D>(d));
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename C, typename D>
      static constexpr auto
      make_constant(C&& c, D&& d)
      {
        return OpenKalman::make_constant<PatternMatrix>(std::forward<C>(c), std::forward<D>(d));
      }


      template<typename S, typename D>
      static constexpr auto
      make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<PatternMatrix, S>(std::forward<D>(d));
      }


      // no get_slice
      // no set_slice
      // no set_triangle
      // no to_diagonal
      // no diagonal_of


      template<typename...Ds, typename Arg>
      static decltype(auto)
      replicate(const std::tuple<Ds...>& tup, Arg&& arg)
      {
        return library_interface<PatternMatrix>::replicate(tup, std::forward<Arg>(arg));
      }


      template<typename...Ds, typename Op, typename...Args>
      static constexpr decltype(auto)
      n_ary_operation(const std::tuple<Ds...>& d_tup, Op&& op, Args&&...args)
      {
        return library_interface<PatternMatrix>::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }


      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        return library_interface<PatternMatrix>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }


      // no to_euclidean
      // no from_euclidean
      // no wrap_angles

      // conjugate is not necessary because it is handled by the general conjugate function.
      // transpose is not necessary because it is handled by the general transpose function.
      // adjoint is not necessary because it is handled by the general adjoint function.
      // determinant is not necessary because it is handled by the general determinant function.


      template<typename A, typename B>
      static constexpr auto sum(A&& a, B&& b)
      {
        return library_interface<PatternMatrix>::sum(std::forward<A>(a), std::forward<B>(b));
      }


      template<typename A, typename B>
      static constexpr auto contract(A&& a, B&& b)
      {
        return library_interface<PatternMatrix>::contract(std::forward<A>(a), std::forward<B>(b));
      }


      // contract_in_place is not necessary because the argument will not be writable.

      // cholesky_factor is not necessary because it is handled by the general cholesky_factor function.


      template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        using Trait = interface::library_interface<PatternMatrix>;
        return Trait::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
      }


      // rank_update_triangular is not necessary because it is handled by the general rank_update_triangular function.

      // solve is not necessary because it is handled by the general solve function.

      // LQ_decomposition is not necessary because it is handled by the general LQ_decomposition function.

      // QR_decomposition is not necessary because it is handled by the general QR_decomposition function.

    };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_CONSTANTADAPTER_HPP
